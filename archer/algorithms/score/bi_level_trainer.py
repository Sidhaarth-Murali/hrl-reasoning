import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
from archer.algorithms.score.rl_guided_trainer import RLGuidedSCoReTrainer
from archer.models.value_function import ValueFunction

class BiLevelSCoReTrainer(RLGuidedSCoReTrainer):
    """
    Bilevel optimization extension of RLGuidedSCoReTrainer.
    
    This trainer adds a value function at the lower level to estimate the expected 
    future reward from the state at turn 2. The value function is used to augment
    the reward signal for training the guidance model (higher level).
    
    Two gradient flow variations are supported:
    1. Full gradient flow: Gradients from both policy and value function flow to the guidance model
    2. Partial gradient flow: Gradients from only the policy flow to the guidance model
       (stop gradients on the value function)
    
    Memory-optimized implementation to avoid OOM issues.
    """
    def __init__(
        self,
        agent,
        tokenizer,
        accelerator,
        guidance_model=None,
        guidance_lr=1e-6,
        guidance_kl_coef=0.05,
        train_guidance_model=True,
        value_model_name="distilroberta-base",
        value_lr=1e-5,
        value_coef=0.5,
        stop_value_gradients=False,
        use_gradient_checkpointing=True,
        cache_dir=None,
        **kwargs
    ):
        """
        Initialize the BiLevelSCoReTrainer with memory optimizations.
        """
        super().__init__(
            agent=agent,
            tokenizer=tokenizer,
            accelerator=accelerator,
            guidance_model=guidance_model,
            guidance_lr=guidance_lr,
            guidance_kl_coef=guidance_kl_coef,
            train_guidance_model=train_guidance_model,
            **kwargs
        )
        
        self.value_coef = value_coef
        self.stop_value_gradients = stop_value_gradients
        
        # Initialize the value function with memory-efficient settings
        self.value_function = ValueFunction(
            model_name=value_model_name,
            device=agent.model.device,
            cache_dir=cache_dir,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        # Create optimizer for the value function
        self.value_optimizer = torch.optim.Adam(
            self.value_function.parameters(),
            lr=value_lr
        )
        
        # Prepare for distributed training
        self.value_function = self.accelerator.prepare(self.value_function)
        self.value_optimizer = self.accelerator.prepare(self.value_optimizer)
        
        print(f"Initialized value function with model {value_model_name}")
        print(f"Value coefficient: {value_coef}")
        print(f"Stop value gradients: {stop_value_gradients}")
        if use_gradient_checkpointing:
            print("Using gradient checkpointing for memory efficiency")

    def train_value_function(self, problems, initial_solutions, guidance_texts, revised_solutions, rewards):
        """
        Train the value function to predict the reward at turn 2.
        Memory-optimized implementation.
        """
        # Convert rewards to tensor
        device = self.agent.model.device
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)
        
        # Process in smaller batches to save memory
        batch_size = len(problems)
        max_batch_size = 16  # Process in smaller batches
        
        total_loss = 0.0
        total_samples = 0
        
        # Clear CUDA cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for i in range(0, batch_size, max_batch_size):
            end_idx = min(i + max_batch_size, batch_size)
            batch_slice = slice(i, end_idx)
            current_batch_size = end_idx - i
            
            # Get value estimates for this batch
            values = self.value_function.get_value(
                problems[batch_slice], 
                initial_solutions[batch_slice], 
                guidance_texts[batch_slice], 
                revised_solutions[batch_slice]
            ).squeeze()
            
            # Calculate MSE loss for this batch
            batch_rewards = rewards_tensor[batch_slice]
            batch_loss = F.mse_loss(values, batch_rewards)
            
            # Scale loss by batch size and accumulate
            scaled_loss = batch_loss * current_batch_size / max_batch_size
            self.accelerator.backward(scaled_loss)
            
            total_loss += batch_loss.item() * current_batch_size
            total_samples += current_batch_size
            
            # Clean up to save memory
            del values, batch_rewards, batch_loss, scaled_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Update the value function after processing all batches
        self.accelerator.clip_grad_norm_(self.value_function.parameters(), self.max_grad_norm)
        self.value_optimizer.step()
        self.value_optimizer.zero_grad()
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        # Calculate value predictions for reporting (use a small subset to save memory)
        report_idx = min(batch_size, 16)  # Only use a few examples for metrics
        with torch.no_grad():
            report_values = self.value_function.get_value(
                problems[:report_idx], 
                initial_solutions[:report_idx], 
                guidance_texts[:report_idx], 
                revised_solutions[:report_idx]
            ).squeeze()
        
        return {
            'value_loss': avg_loss,
            'value_mean': report_values.mean().item(),
            'reward_mean': rewards_tensor[:report_idx].mean().item(),
        }

    def train_guidance(self, trajectories):
        """
        Train the guidance model with the bilevel optimization approach.
        Memory-optimized implementation.
        """
        if not self.train_guidance_model:
            return {}

        # Extract data
        problems = [t[0]['observation'] for t in trajectories]
        initial_solutions = [t[0]['action_turn1'] for t in trajectories]
        revised_solutions = [t[0]['action_turn2'] for t in trajectories]
        raw_rewards = [t[0]['reward_turn2'] for t in trajectories]

        # Generate prompts and hints in smaller batches to save memory
        batch_size = len(problems)
        max_guidance_batch = 32  # Process guidance in smaller batches
        
        analysis_prompts = []
        guidance_texts = []
        
        for i in range(0, batch_size, max_guidance_batch):
            end_idx = min(i + max_guidance_batch, batch_size)
            batch_slice = slice(i, end_idx)
            
            batch_analysis, batch_guidance = self.generate_custom_guidance(
                problems[batch_slice], 
                initial_solutions[batch_slice]
            )
            
            analysis_prompts.extend(batch_analysis)
            guidance_texts.extend(batch_guidance)
            
            # Clean up to save memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        if not guidance_texts:
            return {'guidance_loss': 0.0}

        device = self.agent.model.device
        
        # Symmetric rewards: {0->-1, 1->+1}
        shaped_rewards = [1 if r > 0 else -1 for r in raw_rewards]
        rewards_tensor = torch.tensor(shaped_rewards, dtype=torch.float, device=device)

        # First, train the value function
        value_info = self.train_value_function(
            problems, 
            initial_solutions, 
            guidance_texts, 
            revised_solutions, 
            shaped_rewards
        )

        # Compute policy gradient in batches to save memory
        max_pg_batch = 16  # Process policy gradient in smaller batches
        total_policy_loss = 0.0
        total_samples = 0
        
        self.guidance_optimizer.zero_grad()
        
        for i in range(0, batch_size, max_pg_batch):
            end_idx = min(i + max_pg_batch, batch_size)
            batch_slice = slice(i, end_idx)
            current_batch_size = end_idx - i
            
            # Get value estimates for this batch with appropriate gradient settings
            if self.stop_value_gradients:
                # Variation 2: Stop gradients from the value function to the guidance model
                with torch.no_grad():
                    batch_values = self.value_function.get_value(
                        problems[batch_slice], 
                        initial_solutions[batch_slice], 
                        guidance_texts[batch_slice], 
                        revised_solutions[batch_slice],
                        detach_base_model=True
                    ).squeeze().detach()
            else:
                # Variation 1: Allow gradients from the value function to flow to the guidance model
                batch_values = self.value_function.get_value(
                    problems[batch_slice], 
                    initial_solutions[batch_slice], 
                    guidance_texts[batch_slice], 
                    revised_solutions[batch_slice],
                    detach_base_model=False
                ).squeeze()

            # Combine direct rewards with value estimates
            batch_rewards = rewards_tensor[batch_slice]
            combined_rewards = batch_rewards + self.value_coef * batch_values
            
            # Compute log probabilities for the guidance texts
            batch_prompts = [analysis_prompts[j] for j in range(i, end_idx)]
            batch_texts = [guidance_texts[j] for j in range(i, end_idx)]
            log_probs = self.calculate_guidance_log_probs(batch_prompts, batch_texts)
            
            # Calculate policy gradient loss with the combined rewards
            batch_policy_loss = -torch.mean(log_probs * combined_rewards)
            
            # Scale loss by batch size and accumulate gradients
            scaled_loss = batch_policy_loss * current_batch_size / max_pg_batch
            self.accelerator.backward(scaled_loss)
            
            total_policy_loss += batch_policy_loss.item() * current_batch_size
            total_samples += current_batch_size
            
            # Clean up to save memory
            del batch_values, combined_rewards, log_probs, batch_policy_loss, scaled_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Add KL penalty if configured (compute once for all batches to save computation)
        try:
            # Use a small subset for KL computation to save memory
            kl_subset_size = min(batch_size, 16)
            kl_div = self.calculate_guidance_kl_divergence(
                problems[:kl_subset_size], 
                initial_solutions[:kl_subset_size]
            )
            kl_loss = self.guidance_kl_coef * kl_div
            
            # Apply KL penalty
            self.accelerator.backward(kl_loss)
        except:
            kl_div = torch.tensor(0.0, device=device, requires_grad=True)
            kl_loss = torch.tensor(0.0, device=device)
            
        # Update the guidance model after processing all batches
        self.accelerator.clip_grad_norm_(self.guidance_model.parameters(), self.max_grad_norm)
        self.guidance_optimizer.step()
        
        # Calculate average policy loss
        avg_policy_loss = total_policy_loss / total_samples

        # Return metrics
        info = {
            'guidance_loss': avg_policy_loss,
            'guidance_kl_loss': kl_loss.item() if hasattr(kl_loss, 'item') else 0.0,
            'guidance_total_loss': avg_policy_loss + (kl_loss.item() if hasattr(kl_loss, 'item') else 0.0),
            'guidance_reward_mean': rewards_tensor.mean().item(),
        }
        
        # Add value function metrics
        info.update(value_info)
        
        return info

    def save(self, path):
        """
        Save the models, including the value function.
        Memory-optimized implementation.
        """
        # First clear any cached tensors 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        try:
            # Save the base model and guidance model
            super().save(path)
            
            # Save the value function
            vpath = path.replace('.pt', '_value.pt')
            torch.save({
                'model_state_dict': self.value_function.state_dict(),
                'optimizer_state_dict': self.value_optimizer.state_dict()
            }, vpath)
            print(f"Saved value function to {vpath}")
        except RuntimeError as e:
            print(f"Warning: Memory issue when saving. Error: {e}")
            print("Trying simplified save...")
            
            # Try a simplified save approach to avoid OOM
            if hasattr(self, 'value_function'):
                vpath = path.replace('.pt', '_value.pt')
                # Save state dict directly without optimizer state
                torch.save({
                    'model_state_dict': self.value_function.state_dict(),
                }, vpath)
                print(f"Saved value function model only (no optimizer state) to {vpath}")

    def load(self, path):
        """
        Load the models, including the value function.
        Memory-optimized implementation.
        """
        # First clear any cached tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Load the base model and guidance model
        agent = super().load(path)
        
        # Load the value function if it exists
        vpath = path.replace('.pt', '_value.pt')
        if os.path.exists(vpath):
            try:
                ckpt = torch.load(vpath)
                self.value_function.load_state_dict(ckpt['model_state_dict'])
                if 'optimizer_state_dict' in ckpt:
                    self.value_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print(f"Loaded value function from {vpath}")
            except RuntimeError as e:
                print(f"Warning: Error loading value function: {e}")
                print("Continuing without loading value function optimizer state")
                
        return agent