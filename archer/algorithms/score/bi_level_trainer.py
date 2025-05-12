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
        use_memory_efficient_attention=True,
        max_micro_batch=4,
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
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_memory_efficient_attention=use_memory_efficient_attention,
            max_micro_batch=max_micro_batch,
            **kwargs
        )
        
        self.value_coef = value_coef
        self.stop_value_gradients = stop_value_gradients
        
        # Initialize the value function with memory-efficient settings
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                value_model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            if use_memory_efficient_attention:
                config.use_memory_efficient_attention = True
                config.use_flash_attention = True
            
            self.value_function = ValueFunction(
                model_name=value_model_name,
                device=agent.model.device,
                cache_dir=cache_dir,
                use_gradient_checkpointing=use_gradient_checkpointing,
                config=config
            )
            print("Initialized value function with memory optimizations")
        except Exception as e:
            print(f"Error initializing value function with full optimizations: {e}")
            print("Falling back to basic initialization")
            self.value_function = ValueFunction(
                model_name=value_model_name,
                device=agent.model.device,
                cache_dir=cache_dir,
                use_gradient_checkpointing=use_gradient_checkpointing
            )
        
        # Create optimizer for the value function with gradient accumulation
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
        if use_memory_efficient_attention:
            print("Using memory efficient attention for value function")

    def train_value_function(self, problems, initial_solutions, guidance_texts, revised_solutions, rewards):
        """
        Train the value function to predict the reward at turn 2.
        Memory-optimized implementation with careful cache management.
        """
        # Safe to clear cache before starting
        self.clear_gpu_cache()
        
        # Convert rewards to tensor
        device = self.agent.model.device
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)
        
        # Process in smaller batches to save memory
        batch_size = len(problems)
        max_batch_size = min(32, self.max_micro_batch)  # Use even smaller batches for value function
        
        total_loss = 0.0
        total_samples = 0
        
        # Zero gradients at the start
        self.value_optimizer.zero_grad()
        
        for i in range(0, batch_size, max_batch_size):
            # Clear cache at start of each batch
            self.clear_gpu_cache()
            
            end_idx = min(i + max_batch_size, batch_size)
            batch_slice = slice(i, end_idx)
            current_batch_size = end_idx - i
            
            try:
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
                
                # Backward pass
                self.accelerator.backward(scaled_loss)
                
                total_loss += batch_loss.item() * current_batch_size
                total_samples += current_batch_size
                
                # Only clear cache after backward pass is complete
                # and we've saved any values we need
                del values, batch_rewards, batch_loss, scaled_loss
                self.clear_gpu_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM in batch {i}-{end_idx}, trying with half batch size")
                    # Try again with half the batch size
                    half_size = (end_idx - i) // 2
                    if half_size > 0:
                        # Process first half
                        self.clear_gpu_cache()
                        mid_idx = i + half_size
                        values = self.value_function.get_value(
                            problems[i:mid_idx], 
                            initial_solutions[i:mid_idx], 
                            guidance_texts[i:mid_idx], 
                            revised_solutions[i:mid_idx]
                        ).squeeze()
                        batch_rewards = rewards_tensor[i:mid_idx]
                        batch_loss = F.mse_loss(values, batch_rewards)
                        scaled_loss = batch_loss * half_size / max_batch_size
                        self.accelerator.backward(scaled_loss)
                        total_loss += batch_loss.item() * half_size
                        total_samples += half_size
                        del values, batch_rewards, batch_loss, scaled_loss
                        self.clear_gpu_cache()
                        
                        # Process second half
                        values = self.value_function.get_value(
                            problems[mid_idx:end_idx], 
                            initial_solutions[mid_idx:end_idx], 
                            guidance_texts[mid_idx:end_idx], 
                            revised_solutions[mid_idx:end_idx]
                        ).squeeze()
                        batch_rewards = rewards_tensor[mid_idx:end_idx]
                        batch_loss = F.mse_loss(values, batch_rewards)
                        scaled_loss = batch_loss * half_size / max_batch_size
                        self.accelerator.backward(scaled_loss)
                        total_loss += batch_loss.item() * half_size
                        total_samples += half_size
                        del values, batch_rewards, batch_loss, scaled_loss
                        self.clear_gpu_cache()
                else:
                    raise e
        
        # Update the value function after processing all batches
        self.accelerator.clip_grad_norm_(self.value_function.parameters(), self.max_grad_norm)
        self.value_optimizer.step()
        self.value_optimizer.zero_grad()
        
        # Calculate average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        # Safe to clear cache before new forward pass
        self.clear_gpu_cache()
        
        # Calculate value predictions for reporting (use a small subset to save memory)
        report_idx = min(batch_size, 16)  # Use even smaller subset for reporting
        with torch.no_grad():  # Ensure we don't track gradients for reporting
            report_values = self.value_function.get_value(
                problems[:report_idx], 
                initial_solutions[:report_idx], 
                guidance_texts[:report_idx], 
                revised_solutions[:report_idx]
            ).squeeze()
            
            metrics = {
                'value_loss': avg_loss,
                'value_mean': report_values.mean().item(),
                'reward_mean': rewards_tensor[:report_idx].mean().item(),
            }
        
        # Safe to clear at the end after all computations
        self.clear_gpu_cache()
        
        return metrics

    def train_guidance(self, trajectories):
        """
        Train the guidance model with the bilevel optimization approach.
        Memory-optimized implementation with careful cache management.
        """
        if not self.train_guidance_model:
            return {}

        # Safe to clear cache before starting new training
        self.clear_gpu_cache()

        # Extract data
        problems = [t[0]['observation'] for t in trajectories]
        initial_solutions = [t[0]['action_turn1'] for t in trajectories]
        revised_solutions = [t[0]['action_turn2'] for t in trajectories]
        raw_rewards = [t[0]['reward_turn2'] for t in trajectories]

        # Generate prompts and hints in smaller batches to save memory
        batch_size = len(problems)
        max_guidance_batch = min(32, self.max_micro_batch)  # Use even smaller batches for guidance
        
        analysis_prompts = []
        guidance_texts = []
        
        for i in range(0, batch_size, max_guidance_batch):
            # Clear cache at start of each batch
            self.clear_gpu_cache()
            
            end_idx = min(i + max_guidance_batch, batch_size)
            batch_slice = slice(i, end_idx)
            
            try:
                batch_analysis, batch_guidance = self.generate_custom_guidance(
                    problems[batch_slice], 
                    initial_solutions[batch_slice]
                )
                
                analysis_prompts.extend(batch_analysis)
                guidance_texts.extend(batch_guidance)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM in batch {i}-{end_idx}, trying with half batch size")
                    # Try again with half the batch size
                    half_size = (end_idx - i) // 2
                    if half_size > 0:
                        # Process first half
                        self.clear_gpu_cache()
                        mid_idx = i + half_size
                        batch_analysis, batch_guidance = self.generate_custom_guidance(
                            problems[i:mid_idx], 
                            initial_solutions[i:mid_idx]
                        )
                        analysis_prompts.extend(batch_analysis)
                        guidance_texts.extend(batch_guidance)
                        
                        # Process second half
                        self.clear_gpu_cache()
                        batch_analysis, batch_guidance = self.generate_custom_guidance(
                            problems[mid_idx:end_idx], 
                            initial_solutions[mid_idx:end_idx]
                        )
                        analysis_prompts.extend(batch_analysis)
                        guidance_texts.extend(batch_guidance)
                else:
                    raise e
            
            # Safe to clear after extending lists
            self.clear_gpu_cache()
                
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

        # Safe to clear before policy gradient computation
        self.clear_gpu_cache()

        # Compute policy gradient in batches to save memory
        max_pg_batch = min(16, self.max_micro_batch)  # Use even smaller batches for policy gradient
        total_policy_loss = 0.0
        total_samples = 0
        
        self.guidance_optimizer.zero_grad()
        
        for i in range(0, batch_size, max_pg_batch):
            # Clear cache at start of each batch
            self.clear_gpu_cache()
            
            end_idx = min(i + max_pg_batch, batch_size)
            batch_slice = slice(i, end_idx)
            current_batch_size = end_idx - i
            
            try:
                # Get value estimates for this batch with appropriate gradient settings
                if self.stop_value_gradients:
                    print("🔒 Gradient flow blocked from value function to guidance model.")
                    with torch.no_grad():
                        batch_values = self.value_function.get_value(
                            problems[batch_slice], 
                            initial_solutions[batch_slice], 
                            guidance_texts[batch_slice], 
                            revised_solutions[batch_slice],
                            detach_base_model=True
                        ).squeeze().detach()
                    batch_values = batch_values.detach() 
                else:
                    print("✅ Full gradient flow from both value and policy function to guidance model.")
                    batch_values = self.value_function.get_value(
                        problems[batch_slice], 
                        initial_solutions[batch_slice], 
                        guidance_texts[batch_slice], 
                        revised_solutions[batch_slice],
                        detach_base_model=False
                    ).squeeze()

                # Combine direct rewards with value estimates
                batch_rewards = rewards_tensor[batch_slice]
                batch_rewards = torch.tensor(batch_rewards, dtype=torch.float, device=device, requires_grad=True)
                self.value_coef = torch.tensor(self.value_coef, dtype=torch.float, device=device, requires_grad=True)

                combined_rewards = batch_rewards + self.value_coef * batch_values
                
                # Compute log probabilities for the guidance texts
                batch_prompts = [analysis_prompts[j] for j in range(i, end_idx)]
                batch_texts = [guidance_texts[j] for j in range(i, end_idx)]
                log_probs = self.calculate_guidance_log_probs(batch_prompts, batch_texts)
                log_probs = log_probs.detach().requires_grad_()  

                # Calculate policy gradient loss with the combined rewards
                batch_policy_loss = -torch.mean(log_probs * combined_rewards)
                
                # Scale loss by batch size and accumulate gradients
                scaled_loss = batch_policy_loss * current_batch_size / max_pg_batch
                self.accelerator.backward(scaled_loss)
                
                total_policy_loss += batch_policy_loss.item() * current_batch_size
                total_samples += current_batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM in batch {i}-{end_idx}, trying with half batch size")
                    # Try again with half the batch size
                    half_size = (end_idx - i) // 2
                    if half_size > 0:
                        # Process first half
                        self.clear_gpu_cache()
                        mid_idx = i + half_size
                        self._process_policy_gradient_batch(
                            problems[i:mid_idx],
                            initial_solutions[i:mid_idx],
                            guidance_texts[i:mid_idx],
                            revised_solutions[i:mid_idx],
                            rewards_tensor[i:mid_idx],
                            analysis_prompts[i:mid_idx],
                            guidance_texts[i:mid_idx],
                            half_size,
                            max_pg_batch
                        )
                        
                        # Process second half
                        self.clear_gpu_cache()
                        self._process_policy_gradient_batch(
                            problems[mid_idx:end_idx],
                            initial_solutions[mid_idx:end_idx],
                            guidance_texts[mid_idx:end_idx],
                            revised_solutions[mid_idx:end_idx],
                            rewards_tensor[mid_idx:end_idx],
                            analysis_prompts[mid_idx:end_idx],
                            guidance_texts[mid_idx:end_idx],
                            half_size,
                            max_pg_batch
                        )
                else:
                    raise e
            
            # Only clear cache after backward pass is complete and values are saved
            del batch_values, combined_rewards, log_probs, batch_policy_loss, scaled_loss
            self.clear_gpu_cache()
        
        # Safe to clear before KL computation
        self.clear_gpu_cache()
        
        # Add KL penalty if configured (compute once for all batches to save computation)
        try:
            # Use a small subset for KL computation to save memory
            kl_subset_size = min(batch_size, 16)  # Use even smaller subset for KL
            kl_div = self.calculate_guidance_kl_divergence(
                problems[:kl_subset_size], 
                initial_solutions[:kl_subset_size]
            )
            kl_loss = self.guidance_kl_coef * kl_div
            
            # Apply KL penalty
            self.accelerator.backward(kl_loss)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("OOM during KL computation, skipping KL penalty for this batch")
                kl_div = torch.tensor(0.0, device=device, requires_grad=True)
                kl_loss = torch.tensor(0.0, device=device)
            else:
                raise e
            
        # Update the guidance model after processing all batches
        self.accelerator.clip_grad_norm_(self.guidance_model.parameters(), self.max_grad_norm)
        self.guidance_optimizer.step()
        self.guidance_optimizer.zero_grad()
        
        # Calculate average policy loss
        avg_policy_loss = total_policy_loss / total_samples if total_samples > 0 else 0.0

        # Prepare metrics before final cache clear
        info = {
            'guidance_loss': avg_policy_loss,
            'guidance_kl_loss': kl_loss.item() if hasattr(kl_loss, 'item') else 0.0,
            'guidance_total_loss': avg_policy_loss + (kl_loss.item() if hasattr(kl_loss, 'item') else 0.0),
            'guidance_reward_mean': rewards_tensor.mean().item(),
        }
        
        # Add value function metrics
        info.update(value_info)
        
        # Safe to clear at the end after all computations
        self.clear_gpu_cache()
        
        return info

    def _process_policy_gradient_batch(
        self,
        problems,
        initial_solutions,
        guidance_texts,
        revised_solutions,
        rewards_tensor,
        analysis_prompts,
        guidance_texts_for_probs,
        batch_size,
        max_pg_batch
    ):
        """Helper method to process a single policy gradient batch with memory optimizations."""
        device = self.agent.model.device
        
        # Get value estimates
        if self.stop_value_gradients:
            with torch.no_grad():
                batch_values = self.value_function.get_value(
                    problems, 
                    initial_solutions, 
                    guidance_texts, 
                    revised_solutions,
                    detach_base_model=True
                ).squeeze().detach()
            batch_values = batch_values.detach()
        else:
            batch_values = self.value_function.get_value(
                problems, 
                initial_solutions, 
                guidance_texts, 
                revised_solutions,
                detach_base_model=False
            ).squeeze()

        # Combine rewards
        batch_rewards = torch.tensor(rewards_tensor, dtype=torch.float, device=device, requires_grad=True)
        value_coef = torch.tensor(self.value_coef, dtype=torch.float, device=device, requires_grad=True)
        combined_rewards = batch_rewards + value_coef * batch_values
        
        # Get log probs
        log_probs = self.calculate_guidance_log_probs(analysis_prompts, guidance_texts_for_probs)
        log_probs = log_probs.detach().requires_grad_()
        
        # Calculate and apply loss
        batch_policy_loss = -torch.mean(log_probs * combined_rewards)
        scaled_loss = batch_policy_loss * batch_size / max_pg_batch
        self.accelerator.backward(scaled_loss)
        
        return batch_policy_loss.item() * batch_size

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