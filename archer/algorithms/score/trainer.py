import torch
import transformers
from tqdm import tqdm
from torch.utils.data import DataLoader
from archer.data import DummyDataset
import copy
from typing import Tuple, List, Dict
import random
import time
import torch.nn.functional as F
from archer.prompts import format_math_prompt, format_math_self_correction_prompt

def dict_mean(dict_list):
    """Calculate the mean of values in a list of dictionaries with the same keys."""
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


class SCoReTrainer():
    """
    SCoRe (Self-Correction via Reinforcement Learning) trainer class.
    Implements the two-stage training procedure from Kumar et al. (2024) using pure REINFORCE:
    1. Stage I: Learn to fix mistakes with a strong KL penalty on the first turn
    2. Stage II: Joint multi-turn RL with reward shaping to incentivize actual correction
    
    Uses vanilla policy gradients (REINFORCE) with KL penalty, no critic networks.
    
    This is a FULLY ON-POLICY implementation - no replay buffer is used.
    """
    def __init__(self, 
                 agent,
                 tokenizer,
                 accelerator,
                 lm_lr: float = 5e-6,               # Learning rate for the language model (5e-6 for MATH per paper)
                 grad_accum_steps: int = 8,         # Number of gradient accumulation steps
                 max_grad_norm: float = 1.0,        # Maximum gradient norm for clipping
                 alpha: float = 10.0,               # Reward shaping coefficient (r2 + alpha * (r2 - r1))
                 beta1: float = 0.01,               # KL coefficient for Stage II
                 beta2: float = 0.1,                # KL coefficient for Stage I
                 stage1_steps: int = 1500,          # Number of training steps for Stage I
                 stage2_steps: int = 1500,          # Number of training steps for Stage II
                 batch_size: int = 512):            # Batch size for training (512 for MATH per paper)
        """
        Initialize the SCoRe trainer using pure REINFORCE with KL penalties.
        """
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        
        # Enable gradient checkpointing to save memory
        if hasattr(agent.model, "gradient_checkpointing_enable"):
            agent.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled to save memory")
        
        # Create a frozen reference model for KL divergence calculation
        # This is critical for SCoRe - without this, KL would be zero as we'd compare the model to itself
        print("Creating frozen reference model for KL divergence...")
        with torch.cuda.device(agent.model.device):
            torch.cuda.empty_cache()  # Clear memory before creating reference model
        
        self.ref_model = copy.deepcopy(accelerator.unwrap_model(agent.model)).eval().to(agent.model.device)
        for p in self.ref_model.parameters():
            p.requires_grad = False
        
        print("Reference model created and frozen")
        
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr=lm_lr)
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.lm_optimizer = self.accelerator.prepare(self.lm_optimizer)
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.stage1_steps = stage1_steps
        self.stage2_steps = stage2_steps
        self.current_stage = 1  
        self.total_steps = 0
    
    def calculate_kl_divergence(self, observation, action, reference_model):
        """Calculate KL divergence between current policy and reference policy.
        Computes KL(current || reference) to penalize drift from the reference policy.
        """
        # Tokenize the inputs
        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True).to(self.agent.model.device)
        action_ids = self.tokenizer(action, return_tensors='pt', padding=True).to(self.agent.model.device)
        
        input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim=1)
        attention_mask = torch.cat([obs_ids["attention_mask"], action_ids["attention_mask"]], dim=1)
        
        # Process in smaller chunks to save memory if batch is large
        max_chunk_size = 4  # Process 4 samples at a time
        total_samples = input_ids.size(0)
        kl_div_sum = 0.0
        
        for i in range(0, total_samples, max_chunk_size):
            end_idx = min(i + max_chunk_size, total_samples)
            chunk_input_ids = input_ids[i:end_idx]
            chunk_attention_mask = attention_mask[i:end_idx]
            
            # Get logits from current model WITH gradients enabled
            outputs_current = self.agent.model(input_ids=chunk_input_ids, 
                                             attention_mask=chunk_attention_mask).logits
            
            # Get logits from reference model WITHOUT gradients
            with torch.no_grad():
                outputs_ref = reference_model(input_ids=chunk_input_ids, 
                                            attention_mask=chunk_attention_mask).logits
            
            # Calculate KL divergence for this chunk (KL(current || reference))
            # PyTorch's kl_div takes log-probabilities for input, raw probabilities for target
            # To compute KL(p||q), we need input=log(p), target=q
            log_probs_current = F.log_softmax(outputs_current, dim=-1)
            probs_ref = F.softmax(outputs_ref, dim=-1)
            
            # KL divergence: KL(current || reference)
            chunk_kl_div = F.kl_div(
                input=log_probs_current,
                target=probs_ref,
                reduction='batchmean',
                log_target=False
            )
            
            kl_div_sum += chunk_kl_div * (end_idx - i)
        
        # Average KL divergence
        kl_div = kl_div_sum / total_samples
        
        return kl_div
    
    def stage1_loss(self, observation, action_turn1, action_turn2, reward_turn1, reward_turn2, **kwargs):
        """
        Stage I loss: optimize for second turn reward with strong KL penalty on first turn.
        Loss = -r2 + beta2 * KL(pi(·|turn1) || ref(·|turn1))
        
        Uses pure REINFORCE (policy gradients) with no critic networks.
        """

        # Calculate KL divergence for first turn using the frozen reference model
        kl_div = self.calculate_kl_divergence(observation, action_turn1, self.ref_model)
        
        # Calculate log probabilities for second turn using REINFORCE
        prompts = [format_math_self_correction_prompt(o + a1) for o, a1 in zip(observation, action_turn1)]
        log_prob_turn2 = self.agent.get_log_prob(prompts, action_turn2)
        
        # Convert rewards to tensors (avoiding the UserWarning by properly using .clone().detach())
        reward_turn2_tensor = torch.as_tensor(reward_turn2, device=self.agent.model.device, dtype=torch.float).detach()
        
        # Ensure log_prob has proper shape
        if log_prob_turn2.dim() > 1:
            log_prob_turn2 = log_prob_turn2.flatten()
            
        # Calculate Stage I loss: -r2 + beta2 * KL
        policy_loss = -torch.mean(log_prob_turn2 * reward_turn2_tensor)
        reg_loss = self.beta2 * kl_div
        loss = policy_loss + reg_loss
        
        # Save metric values before cleaning up tensors
        policy_loss_value = policy_loss.detach().cpu().item()
        reg_loss_value = reg_loss.detach().cpu().item()
        loss_value = loss.detach().cpu().item()
        kl_div_value = kl_div.detach().cpu().item()
        reward_turn2_mean = sum(reward_turn2) / len(reward_turn2)        
        # Backward pass with memory cleanup
        self.accelerator.backward(loss)
            
        return {
            "stage1_loss": loss_value,
            "stage1_policy_loss": policy_loss_value,
            "stage1_kl_loss": reg_loss_value,
            "stage1_reward_turn2_mean": reward_turn2_mean,
            "stage1_kl_div": kl_div_value
        }
    
    def stage2_loss(self, observation, action_turn1, action_turn2, reward_turn1, reward_turn2, **kwargs):
        """
        Stage II loss: jointly optimize both turns with reward shaping.
        r2_shaped = r2 + alpha * (r2 - r1)
        Loss = -(r1 + r2_shaped) + beta1 * (KL1 + KL2)
        
        Uses pure REINFORCE (policy gradients) with no critic networks.
        """        
        # Calculate KL divergence for both turns using the frozen reference model
        kl_div_turn1 = self.calculate_kl_divergence(observation, action_turn1, self.ref_model)
        
        correction_prompts = [format_math_self_correction_prompt(o + a1) for o, a1 in zip(observation, action_turn1)]
        kl_div_turn2 = self.calculate_kl_divergence(correction_prompts, action_turn2, self.ref_model)
        
        total_kl = kl_div_turn1 + kl_div_turn2
        
        # Calculate log probabilities for both turns (REINFORCE)
        log_prob_turn1 = self.agent.get_log_prob(observation, action_turn1)
        log_prob_turn2 = self.agent.get_log_prob(correction_prompts, action_turn2)
        
        # Convert rewards to tensors (with proper as_tensor to avoid memory issues)
        reward_turn1_tensor = torch.as_tensor(reward_turn1, device=self.agent.model.device, dtype=torch.float).detach()
        reward_turn2_tensor = torch.as_tensor(reward_turn2, device=self.agent.model.device, dtype=torch.float).detach()
        
        # Apply reward shaping to turn 2: r2 + alpha * (r2 - r1)
        shaped_reward_turn2 = reward_turn2_tensor + self.alpha * (reward_turn2_tensor - reward_turn1_tensor)
        
        # Ensure log probs have proper shape
        if log_prob_turn1.dim() > 1:
            log_prob_turn1 = log_prob_turn1.flatten()
        if log_prob_turn2.dim() > 1:
            log_prob_turn2 = log_prob_turn2.flatten()
        
        # Calculate components of loss
        policy_loss_turn1 = torch.mean(log_prob_turn1 * reward_turn1_tensor)
        policy_loss_turn2 = torch.mean(log_prob_turn2 * shaped_reward_turn2)
        reg_loss = self.beta1 * total_kl
        
        # Final loss
        loss = -(policy_loss_turn1 + policy_loss_turn2) + reg_loss
        
        # Save metrics before cleaning up
        shaped_reward_mean = shaped_reward_turn2.detach().cpu().mean().item()
        loss_value = loss.detach().cpu().item()
        kl_value = total_kl.detach().cpu().item()
        reward_turn1_mean = sum(reward_turn1) / len(reward_turn1)
        reward_turn2_mean = sum(reward_turn2) / len(reward_turn2)
                
        # Backward pass
        self.accelerator.backward(loss)
        return {
            "stage2_loss": loss_value,
            "stage2_reward_turn1_mean": reward_turn1_mean,
            "stage2_reward_turn2_mean": reward_turn2_mean,
            "stage2_shaped_reward_mean": shaped_reward_mean,
            "stage2_kl_div": kl_value
        }
    
    def train_stage1(self, trajectories):
        """Train in Stage I: focus on second-turn reward with KL constraint on first turn.
           Fully on-policy: uses the trajectories that were just collected."""
        print(">>> SCoRe Stage I Training (Pure REINFORCE with KL)")
        info_list = []
        
        data = self.process_trajectories(trajectories)
        actual_batch_size = min(self.batch_size, len(data), 2) 
        dataloader = DataLoader(DummyDataset(data), batch_size=actual_batch_size, shuffle=True)
        dataloader = self.accelerator.prepare(dataloader)
        
        # Update with pure policy gradients (REINFORCE)
        self.lm_optimizer.zero_grad()
        for batch in tqdm(dataloader, desc="Stage I Update", leave=False):
            # Process batch and add to info list
            batch_info = self.stage1_loss(**batch)
            info_list.append(batch_info)
            
            # Periodically free memory even during batch processing
            if len(info_list) % 2 == 0:
                with torch.cuda.device(self.agent.model.device):
                    torch.cuda.empty_cache()
        
        # Apply gradient clipping and step
        self.accelerator.clip_grad_norm_(self.agent.model.parameters(), self.max_grad_norm)
        self.lm_optimizer.step()        
        self.total_steps += 1

        # Calculate mean metrics across all batches
        result = dict_mean(info_list)
        
        return result
    
    def train_stage2(self, trajectories):
        """Train in Stage II: joint multi-turn RL with reward shaping.
           Fully on-policy: uses the trajectories that were just collected."""
        print(">>> SCoRe Stage II Training (Pure REINFORCE with KL & Reward Shaping)")
        info_list = []
        

        
        # Process trajectories into a format suitable for training
        data = self.process_trajectories(trajectories)
        
        # Use smaller batch size if needed
        actual_batch_size = min(self.batch_size, len(data), 2)  # Further limit batch size to 8 max
        dataloader = DataLoader(DummyDataset(data), batch_size=actual_batch_size, shuffle=True)
        dataloader = self.accelerator.prepare(dataloader)
        
        # Update with pure policy gradients (REINFORCE)
        self.lm_optimizer.zero_grad()
        
        for batch in tqdm(dataloader, desc="Stage II Update", leave=False):
            # Process batch and add to info list
            batch_info = self.stage2_loss(**batch)
            info_list.append(batch_info)
            
            # Periodically free memory even during batch processing
            if len(info_list) % 2 == 0:
                with torch.cuda.device(self.agent.model.device):
                    torch.cuda.empty_cache()
        
        # Apply gradient clipping and step
        self.accelerator.clip_grad_norm_(self.agent.model.parameters(), self.max_grad_norm)
        self.lm_optimizer.step()

        
        self.total_steps += 1
    
        # Check if we've completed Stage II
        if self.total_steps >= self.stage1_steps + self.stage2_steps:
            print(f">>> Completed Stage II after total {self.total_steps} steps")
            # Save final checkpoint
            self.save_stage_checkpoint(2)
            print("SCoRe training complete! The model has finished both stages of training.")

        # Calculate mean metrics across all batches
        result = dict_mean(info_list)
        
        return result
    
    def process_trajectories(self, trajectories):
        """Process trajectories into a format suitable for training.
        This is on-policy - using trajectories collected in the current iteration."""
        processed_data = []
        
        # Process each trajectory
        for traj in trajectories:
            # Extract data from the trajectory
            processed_data.append({
                "observation": traj[0]["observation"],
                "action_turn1": traj[0]["action_turn1"],
                "action_turn2": traj[0]["action_turn2"],
                "reward_turn1": traj[0]["reward_turn1"],
                "reward_turn2": traj[0]["reward_turn2"]
            })
        
        return processed_data
    
    def update(self, trajectories, no_update_actor=False):
        """Main update function called from the training loop.
           Takes trajectories directly instead of sampling from a replay buffer."""
        info = {}
        breakpoint()
        # Check if we should transition to Stage II
        if self.current_stage == 1 and self.total_steps >= self.stage1_steps:
            self.current_stage = 2
            print(f">>> Transitioning to Stage II after {self.total_steps} steps")
            
            # Save Stage I checkpoint
            self.save_stage_checkpoint(1)
        
        # No critic updates - pure REINFORCE
        if not no_update_actor:
            if self.current_stage == 1:
                actor_info = self.train_stage1(trajectories)
            else:  # Stage 2
                actor_info = self.train_stage2(trajectories)
            
            info.update(actor_info)
        
        return info
    
    def save_stage_checkpoint(self, stage_num):
        """Save a checkpoint at the end of a training stage."""
        try:
            checkpoint_path = f"/home/pramit/hrl-nips-work/hrl-reasoning/.saved_models/score_stage{stage_num}_checkpoint.pt"
            self.save(checkpoint_path)
            print(f"Saved Stage {stage_num} checkpoint to {checkpoint_path}")
        except Exception as e:
            print(f"Failed to save Stage {stage_num} checkpoint: {e}")
    
    def save(self, path):
        """Save the model state."""
        torch.save({
            'model_state_dict': self.accelerator.unwrap_model(self.agent.model).state_dict(),
            'lm_optimizer_state_dict': self.lm_optimizer.state_dict(),
            'current_stage': self.current_stage,
            'total_steps': self.total_steps
        }, path)
    
    def load(self, path):
        """Load the model state."""
        checkpoint = torch.load(path)
        self.agent.model.load_state_dict(checkpoint['model_state_dict'])
        self.lm_optimizer.load_state_dict(checkpoint['lm_optimizer_state_dict'])
        
        # Load SCoRe-specific state variables if available
        if 'current_stage' in checkpoint:
            self.current_stage = checkpoint['current_stage']
        if 'total_steps' in checkpoint:
            self.total_steps = checkpoint['total_steps']
        
        return self.agent