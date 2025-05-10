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
    Memory-optimized implementation to handle larger models.
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
                 batch_size: int = 128,             # Batch size for training (512 for MATH per paper)
                 max_micro_batch: int = 8,          # Maximum size of micro-batches for memory efficiency
                 use_gradient_checkpointing: bool = True,  # Whether to use gradient checkpointing
                 use_memory_efficient_attention: bool = True):  # Whether to use memory efficient attention
        """
        Initialize the SCoRe trainer using pure REINFORCE with KL penalties.
        Memory-optimized implementation.
        """
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        self.max_micro_batch = max_micro_batch
        
        # Enable memory optimizations
        if use_gradient_checkpointing and hasattr(agent.model, "gradient_checkpointing_enable"):
            agent.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled to save memory")
            
        if use_memory_efficient_attention and hasattr(agent.model, "use_memory_efficient_attention"):
            agent.model.use_memory_efficient_attention = True
            print("Memory efficient attention enabled")
        
        # Create a frozen reference model for KL divergence calculation
        print("Creating frozen reference model for KL divergence...")
        with torch.cuda.device(agent.model.device):
            torch.cuda.empty_cache()  # Clear memory before creating reference model
        
        self.ref_model = copy.deepcopy(accelerator.unwrap_model(agent.model)).eval().to(agent.model.device)
        for p in self.ref_model.parameters():
            p.requires_grad = False
        
        print("Reference model created and frozen")
        
        # Initialize optimizer with memory-efficient settings
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
        
        # Enable memory-efficient attention for reference model if available
        if use_memory_efficient_attention and hasattr(self.ref_model, "use_memory_efficient_attention"):
            self.ref_model.use_memory_efficient_attention = True

    def calculate_kl_divergence(self, observation, action, reference_model):
        """Calculate KL divergence between current policy and reference policy.
        Computes KL(current || reference) to penalize drift from the reference policy.
        Memory-optimized implementation with improved chunking and cleanup.
        """
        # Process in very small chunks to minimize memory usage
        max_chunk_size = 4  # Reduced from previous value for more granular processing
        total_samples = len(observation) if isinstance(observation, list) else 1
        kl_div_sum = 0.0
        total_tokens = 0
        
        # Convert single inputs to lists for consistent processing
        if not isinstance(observation, list):
            observation = [observation]
            action = [action]
            
        for i in range(0, total_samples, max_chunk_size):
            end_idx = min(i + max_chunk_size, total_samples)
            chunk_obs = observation[i:end_idx]
            chunk_act = action[i:end_idx]
            
            # Tokenize the inputs for this small chunk
            obs_ids = self.tokenizer(chunk_obs, return_tensors='pt', padding=True, truncation=True, max_length=512)
            act_ids = self.tokenizer(chunk_act, return_tensors='pt', padding=True, truncation=True, max_length=512)
            
            # Move to device
            obs_ids = {k: v.to(self.agent.model.device) for k, v in obs_ids.items()}
            act_ids = {k: v.to(self.agent.model.device) for k, v in act_ids.items()}
            
            input_ids = torch.cat([obs_ids["input_ids"], act_ids["input_ids"]], dim=1)
            attention_mask = torch.cat([obs_ids["attention_mask"], act_ids["attention_mask"]], dim=1)
            
            # Process tokens in smaller sub-chunks if sequence is long
            seq_length = input_ids.size(1)
            max_seq_chunk = 128  # Process sequence in smaller chunks
            
            chunk_kl_div = 0.0
            chunk_tokens = 0
            
            for seq_start in range(0, seq_length, max_seq_chunk):
                seq_end = min(seq_start + max_seq_chunk, seq_length)
                
                # Get sequence chunk
                chunk_input_ids = input_ids[:, seq_start:seq_end]
                chunk_attention_mask = attention_mask[:, seq_start:seq_end]
                
                # Get logits from current model WITH gradients enabled
                outputs_current = self.agent.model(
                    input_ids=chunk_input_ids,
                    attention_mask=chunk_attention_mask
                ).logits
                
                # Get logits from reference model WITHOUT gradients
                with torch.no_grad():
                    outputs_ref = reference_model(
                        input_ids=chunk_input_ids,
                        attention_mask=chunk_attention_mask
                    ).logits
                
                # Calculate KL divergence for this sub-chunk
                log_probs_current = F.log_softmax(outputs_current, dim=-1)
                probs_ref = F.softmax(outputs_ref, dim=-1)
                
                # Compute KL divergence with memory-efficient operations
                sub_chunk_kl = F.kl_div(
                    input=log_probs_current.float(),  # Ensure float32
                    target=probs_ref.float(),        # Ensure float32
                    reduction='none',                # No reduction to save memory
                    log_target=False
                )
                
                # Sum over vocabulary dimension first
                sub_chunk_kl = sub_chunk_kl.sum(dim=-1)  # Sum over vocab
                
                # Apply attention mask and get mean
                valid_tokens = chunk_attention_mask.sum().item()
                sub_chunk_kl = (sub_chunk_kl * chunk_attention_mask).sum() / valid_tokens
                
                chunk_kl_div += sub_chunk_kl.item()
                chunk_tokens += valid_tokens
                
                # Clean up GPU memory
                del outputs_current, outputs_ref, log_probs_current, probs_ref, sub_chunk_kl
                torch.cuda.empty_cache()
            
            # Accumulate KL divergence weighted by number of tokens
            kl_div_sum += chunk_kl_div * chunk_tokens
            total_tokens += chunk_tokens
            
            # Clean up chunk tensors
            del obs_ids, act_ids, input_ids, attention_mask
            torch.cuda.empty_cache()
        
        # Return average KL divergence across all tokens
        return torch.tensor(kl_div_sum / total_tokens, device=self.agent.model.device)
    
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
        Memory-optimized implementation to handle larger models.
        """
        # Process in smaller micro-batches to save memory
        batch_size = len(observation)
        max_micro_batch = min(32, batch_size)  # Process at most 32 examples at once
        
        # Initialize accumulators
        total_loss = 0.0
        total_kl = 0.0
        total_policy_loss_turn1 = 0.0
        total_policy_loss_turn2 = 0.0
        
        # Track metrics for reporting
        all_shaped_rewards = []
        
        # Clear CUDA cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Convert rewards to tensors once (with proper device placement)
        device = self.agent.model.device
        reward_turn1_tensor = torch.tensor(reward_turn1, device=device, dtype=torch.float32)
        reward_turn2_tensor = torch.tensor(reward_turn2, device=device, dtype=torch.float32)
        
        # Apply reward shaping to turn 2: r2 + alpha * (r2 - r1)
        shaped_reward_turn2 = reward_turn2_tensor + self.alpha * (reward_turn2_tensor - reward_turn1_tensor)
        all_shaped_rewards = shaped_reward_turn2.detach().cpu().tolist()
        
        # For each micro-batch
        for i in range(0, batch_size, max_micro_batch):
            end_idx = min(i + max_micro_batch, batch_size)
            batch_slice = slice(i, end_idx)
            current_batch_size = end_idx - i
            
            # Get micro-batch data
            obs_batch = [observation[j] for j in range(i, end_idx)]
            act1_batch = [action_turn1[j] for j in range(i, end_idx)]
            act2_batch = [action_turn2[j] for j in range(i, end_idx)]
            r1_batch = reward_turn1_tensor[batch_slice]
            shaped_r2_batch = shaped_reward_turn2[batch_slice]
            
            # Create correction prompts
            correction_prompts = [format_math_self_correction_prompt(o + a1) for o, a1 in zip(obs_batch, act1_batch)]
            
            # Calculate KL divergence for turn 1 (with memory cleanup)
            kl_div_turn1 = self.calculate_kl_divergence(obs_batch, act1_batch, self.ref_model)
            
            # Calculate KL divergence for turn 2 (with memory cleanup)
            kl_div_turn2 = self.calculate_kl_divergence(correction_prompts, act2_batch, self.ref_model)
            
            batch_kl = kl_div_turn1 + kl_div_turn2
            total_kl += batch_kl.item() * current_batch_size
            
            # Calculate log probabilities for both turns (REINFORCE)
            log_prob_turn1 = self.agent.get_log_prob(obs_batch, act1_batch)
            if log_prob_turn1.dim() > 1:
                log_prob_turn1 = log_prob_turn1.flatten()
                
            log_prob_turn2 = self.agent.get_log_prob(correction_prompts, act2_batch)
            if log_prob_turn2.dim() > 1:
                log_prob_turn2 = log_prob_turn2.flatten()
            
            # Calculate components of loss
            policy_loss_turn1 = torch.mean(log_prob_turn1 * r1_batch)
            policy_loss_turn2 = torch.mean(log_prob_turn2 * shaped_r2_batch)
            
            # Scale the batch KL regulation by batch size
            reg_loss = self.beta1 * batch_kl * (current_batch_size / batch_size)
            
            # Final loss for this micro-batch
            batch_loss = -(policy_loss_turn1 + policy_loss_turn2) + reg_loss
            
            # Accumulate metrics
            total_policy_loss_turn1 += policy_loss_turn1.item() * current_batch_size
            total_policy_loss_turn2 += policy_loss_turn2.item() * current_batch_size
            total_loss += batch_loss.item() * current_batch_size
            
            # Backward pass with scaling
            scaled_loss = batch_loss * (current_batch_size / max_micro_batch)
            self.accelerator.backward(scaled_loss)
            
            # Clean up tensors to free memory
            del obs_batch, act1_batch, act2_batch, r1_batch, shaped_r2_batch
            del correction_prompts, log_prob_turn1, log_prob_turn2
            del batch_kl, policy_loss_turn1, policy_loss_turn2, reg_loss, batch_loss, scaled_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate final metrics
        avg_loss = total_loss / batch_size
        avg_kl = total_kl / batch_size
        avg_policy_loss_turn1 = total_policy_loss_turn1 / batch_size
        avg_policy_loss_turn2 = total_policy_loss_turn2 / batch_size
        
        reward_turn1_mean = sum(reward_turn1) / len(reward_turn1)
        reward_turn2_mean = sum(reward_turn2) / len(reward_turn2)
        shaped_reward_mean = sum(all_shaped_rewards) / len(all_shaped_rewards)
        
        return {
            "stage2_loss": avg_loss,
            "stage2_reward_turn1_mean": reward_turn1_mean,
            "stage2_reward_turn2_mean": reward_turn2_mean,
            "stage2_shaped_reward_mean": shaped_reward_mean,
            "stage2_kl_div": avg_kl
        }
    
    def clear_gpu_cache(self):
        """Helper method to clear GPU cache and force garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()

    def train_stage1(self, trajectories):
        """Train in Stage I: focus on second-turn reward with KL constraint on first turn.
           Fully on-policy: uses the trajectories that were just collected.
           Memory-optimized implementation."""
        print(">>> SCoRe Stage I Training (Pure REINFORCE with KL)")
        info_list = []
        
        # Clear cache before training
        self.clear_gpu_cache()
        
        data = self.process_trajectories(trajectories)
        actual_batch_size = min(self.batch_size, len(data), self.max_micro_batch) 
        dataloader = DataLoader(DummyDataset(data), batch_size=actual_batch_size, shuffle=True)
        dataloader = self.accelerator.prepare(dataloader)
        
        # Update with pure policy gradients (REINFORCE)
        self.lm_optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Stage I Update", leave=False)):
            # Process batch and add to info list
            batch_info = self.stage1_loss(**batch)
            info_list.append(batch_info)
            
            # Accumulate gradients
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.accelerator.clip_grad_norm_(self.agent.model.parameters(), self.max_grad_norm)
                self.lm_optimizer.step()
                self.lm_optimizer.zero_grad()
                # Clear cache after optimizer step
                self.clear_gpu_cache()
            
            # Periodically free memory
            if batch_idx % 2 == 0:
                self.clear_gpu_cache()
        
        # Handle remaining gradients
        if len(dataloader) % self.grad_accum_steps != 0:
            self.accelerator.clip_grad_norm_(self.agent.model.parameters(), self.max_grad_norm)
            self.lm_optimizer.step()
            self.lm_optimizer.zero_grad()
            self.clear_gpu_cache()
        
        self.total_steps += 1

        # Calculate mean metrics across all batches
        result = dict_mean(info_list)
        
        # Final cache clear
        self.clear_gpu_cache()
        return result
    
    def train_stage2(self, trajectories):
        """Train in Stage II: joint multi-turn RL with reward shaping.
           Fully on-policy: uses the trajectories that were just collected.
           Memory-optimized implementation."""
        print(">>> SCoRe Stage II Training (Pure REINFORCE with KL & Reward Shaping)")
        info_list = []
        
        # Clear cache before training
        self.clear_gpu_cache()
        
        # Process trajectories into a format suitable for training
        data = self.process_trajectories(trajectories)
        
        # Use smaller batch size for memory efficiency
        actual_batch_size = min(self.batch_size, len(data), self.max_micro_batch)
        dataloader = DataLoader(DummyDataset(data), batch_size=actual_batch_size, shuffle=True)
        dataloader = self.accelerator.prepare(dataloader)
        
        # Update with pure policy gradients (REINFORCE)
        self.lm_optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Stage II Update", leave=False)):
            # Process batch and add to info list
            batch_info = self.stage2_loss(**batch)
            info_list.append(batch_info)
            
            # Accumulate gradients
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.accelerator.clip_grad_norm_(self.agent.model.parameters(), self.max_grad_norm)
                self.lm_optimizer.step()
                self.lm_optimizer.zero_grad()
                # Clear cache after optimizer step
                self.clear_gpu_cache()
            
            # Periodically free memory
            if batch_idx % 2 == 0:
                self.clear_gpu_cache()
        
        # Handle remaining gradients
        if len(dataloader) % self.grad_accum_steps != 0:
            self.accelerator.clip_grad_norm_(self.agent.model.parameters(), self.max_grad_norm)
            self.lm_optimizer.step()
            self.lm_optimizer.zero_grad()
            self.clear_gpu_cache()
        
        self.total_steps += 1
    
        # Check if we've completed Stage II
        if self.total_steps >= self.stage1_steps + self.stage2_steps:
            print(f">>> Completed Stage II after total {self.total_steps} steps")
            # Save final checkpoint
            self.save_stage_checkpoint(2)
            print("SCoRe training complete! The model has finished both stages of training.")

        # Calculate mean metrics across all batches
        result = dict_mean(info_list)
        
        # Final cache clear
        self.clear_gpu_cache()
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
        # Clear cache before saving
        self.clear_gpu_cache()
        
        torch.save({
            'model_state_dict': self.accelerator.unwrap_model(self.agent.model).state_dict(),
            'lm_optimizer_state_dict': self.lm_optimizer.state_dict(),
            'current_stage': self.current_stage,
            'total_steps': self.total_steps
        }, path)
        
        # Clear cache after saving
        self.clear_gpu_cache()
    
    def load(self, path):
        """Load the model state."""
        # Clear cache before loading
        self.clear_gpu_cache()
        
        checkpoint = torch.load(path)
        self.agent.model.load_state_dict(checkpoint['model_state_dict'])
        self.lm_optimizer.load_state_dict(checkpoint['lm_optimizer_state_dict'])
        
        # Load SCoRe-specific state variables if available
        if 'current_stage' in checkpoint:
            self.current_stage = checkpoint['current_stage']
        if 'total_steps' in checkpoint:
            self.total_steps = checkpoint['total_steps']
            
        # Clear cache after loading
        self.clear_gpu_cache()
        return self.agent