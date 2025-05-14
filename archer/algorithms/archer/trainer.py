import torch
import transformers
from tqdm import tqdm
from torch.utils.data import DataLoader
from archer.data import DummyDataset
import copy
import threading
from typing import Tuple
import random
import time
import gc
import os

def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

class ArcherTrainer():
    def __init__(self, agent,\
                 accelerator,\
                    tokenizer,\
                    critic_lr: float = 1e-3,\
                    lm_lr: float = 1e-5,\
                    grad_accum_steps: int = 8,\
                    gamma: float = 0.9,
                    tau: float = 0.1,
                    epochs: int = 3,
                    max_grad_norm: float=0.01,
                    actor_epochs: int = 3,
                    max_micro_batch: int = 4,  # Maximum size of micro-batches for memory efficiency
                    use_gradient_checkpointing: bool = True,  # Whether to use gradient checkpointing
                    use_memory_efficient_attention: bool = True):  # Whether to use memory efficient attention
        """
        beta: coefficient for the bc loss
        
        Args:
            agent: The agent to train
            accelerator: The Accelerate accelerator
            tokenizer: The tokenizer
            critic_lr: Learning rate for the critic
            lm_lr: Learning rate for the language model
            grad_accum_steps: Number of gradient accumulation steps
            gamma: Discount factor
            tau: Target network update rate
            epochs: Number of epochs to train the critic
            max_grad_norm: Maximum gradient norm
            actor_epochs: Number of epochs to train the actor
            max_micro_batch: Maximum size of micro-batches for memory efficiency
            use_gradient_checkpointing: Whether to use gradient checkpointing
            use_memory_efficient_attention: Whether to use memory efficient attention
        """
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr=lm_lr)
        self.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=critic_lr)
        self.criterion = torch.nn.MSELoss()
        self.grad_accum_steps = grad_accum_steps
        self.actor_epochs = actor_epochs
        self.gamma = gamma
        self.epochs = epochs
        self.step = 0
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.max_micro_batch = max_micro_batch
        
        # Enable memory optimizations
        if use_gradient_checkpointing:
            if hasattr(agent.model, "gradient_checkpointing_enable"):
                agent.model.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled for model")
            if hasattr(agent.critic, "gradient_checkpointing_enable"):
                agent.critic.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled for critic")
        
        if use_memory_efficient_attention:
            # Enable memory efficient attention for model
            if hasattr(agent.model, "config") and hasattr(agent.model.config, "use_memory_efficient_attention"):
                agent.model.config.use_memory_efficient_attention = True
                print("Memory efficient attention enabled for model")
            
            # Enable flash attention if available
            if hasattr(agent.model, "config") and hasattr(agent.model.config, "use_flash_attention"):
                agent.model.config.use_flash_attention = True
                print("Flash attention enabled for model")
        
        # Verify all models are on the correct device after preparing with accelerator
        self.critic_optimizer, self.lm_optimizer = self.accelerator.prepare(self.critic_optimizer, self.lm_optimizer)
        self.agent.model = self.accelerator.prepare(self.agent.model)
        self.agent.critic = self.accelerator.prepare(self.agent.critic)
        self.agent.target_critic = self.accelerator.prepare(self.agent.target_critic)
        
        # Print device information for debugging
        device = self.accelerator.device
        print(f"Using device: {device}")
        print(f"Model device: {next(self.agent.model.parameters()).device}")
        print(f"Critic device: {next(self.agent.critic.parameters()).device}")
        print(f"Target critic device: {next(self.agent.target_critic.parameters()).device}")
    
    def clear_gpu_cache(self):
        """Helper to clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def to_tensor(self, data, flatten=False):
        """Helper to convert data to tensor and move to the right device."""
        if isinstance(data, torch.Tensor):
            tensor = data
        else:
            tensor = torch.tensor(data, dtype=self.accelerator.unwrap_model(self.agent.model).dtype)
        
        tensor = tensor.to(self.accelerator.device)
        if flatten:
            tensor = tensor.flatten()
        return tensor

    def critic_loss(self, observation, action, reward, next_observation, done, mc_return,**kwargs):
        # Convert inputs to tensors and move to device
        reward = torch.Tensor(reward).to(self.accelerator.unwrap_model(self.agent.model).device, 
                                          dtype=self.accelerator.unwrap_model(self.agent.model).dtype).flatten()
        done = torch.Tensor(done).to(self.accelerator.unwrap_model(self.agent.model).device, 
                                      dtype=torch.float).flatten()
        
        try:
            # Get critic outputs
            q1, q2, v1, v2 = self.agent.critic(observation, action, detach_model=False)
            
            # Get target values
            with torch.no_grad():
                pi_action = self.agent.get_action(copy.deepcopy(observation))
                target_q1, target_q2, _, _ = self.agent.target_critic(copy.deepcopy(observation), pi_action, detach_model=False)
                
                # Get next state values from target critic
                _, _, target_v1, target_v2 = self.agent.target_critic(next_observation, copy.deepcopy(action))
                target_v1 = reward + (1 - done) * target_v1.flatten() * self.gamma
                target_v2 = reward + (1 - done) * target_v2.flatten() * self.gamma
            
            # Flatten critic outputs
            q1 = q1.flatten()
            q2 = q2.flatten()
            v1 = v1.flatten()
            v2 = v2.flatten()
            target_q1 = target_q1.flatten()
            target_q2 = target_q2.flatten()
            
            # Calculate losses
            q1_loss = self.criterion(q1, target_v1)
            q2_loss = self.criterion(q2, target_v2)
            v1_loss = self.criterion(v1, target_q1)
            v2_loss = self.criterion(v2, target_q2)
            
            # Backward pass
            self.accelerator.backward((q1_loss + q2_loss + v1_loss + v2_loss))
            
            # Detach tensors and move to CPU to save memory
            q1_loss, q2_loss = q1_loss.detach().cpu(), q2_loss.detach().cpu()
            v1_loss, v2_loss = v1_loss.detach().cpu(), v2_loss.detach().cpu()
            q1, q2 = q1.detach().cpu(), q2.detach().cpu()
            v1, v2 = v1.detach().cpu(), v2.detach().cpu()
            target_q1, target_q2 = target_q1.detach().cpu(), target_q2.detach().cpu()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM error in critic loss: {e}")
                self.clear_gpu_cache()
                # Return empty dict with zeros to prevent training failure
                return {
                    "q1.loss": 0.0, "q2.loss": 0.0, "v1.loss": 0.0, "v2.loss": 0.0,
                    "q1.mean": 0.0, "q1.min": 0.0, "q1.max": 0.0, "q1.std": 0.0,
                    "q2.mean": 0.0, "q2.max": 0.0, "q2.min": 0.0, "q2.std": 0.0,
                    "v1.mean": 0.0, "v1.min": 0.0, "v1.max": 0.0, "v1.std": 0.0,
                    "v2.mean": 0.0, "v2.max": 0.0, "v2.min": 0.0, "v2.std": 0.0,
                    "target_q1.mean": 0.0, "target_q1.min": 0.0, "target_q1.max": 0.0, "target_q1.std": 0.0,
                    "target_q2.mean": 0.0, "target_q2.max": 0.0, "target_q2.min": 0.0, "target_q2.std": 0.0,
                }
            else:
                raise e
        
        # Clear any references to intermediate tensors
        del pi_action, target_v1, target_v2
        self.clear_gpu_cache()
            
        return {
            "q1.loss": q1_loss, "q2.loss": q2_loss, "v1.loss": v1_loss, "v2.loss": v2_loss,
            "q1.mean": torch.mean(q1), "q1.min": torch.min(q1), "q1.max": torch.max(q1), "q1.std": torch.std(q1),
            "q2.mean": torch.mean(q2), "q2.max": torch.max(q2), "q2.min": torch.min(q2), "q2.std": torch.std(q2),
            "v1.mean": torch.mean(v1), "v1.min": torch.min(v1), "v1.max": torch.max(v1), "v1.std": torch.std(v1),
            "v2.mean": torch.mean(v2), "v2.max": torch.max(v2), "v2.min": torch.min(v2), "v2.std": torch.std(v2),
            "target_q1.mean": torch.mean(target_q1), "target_q1.min": torch.min(target_q1),
            "target_q1.max": torch.max(target_q1), "target_q1.std": torch.std(target_q1),
            "target_q2.mean": torch.mean(target_q2), "target_q2.max": torch.max(target_q2),
            "target_q2.min": torch.min(target_q2), "target_q2.std": torch.std(target_q2),
        }

    def actor_loss(self, observation, pi_action, advantage, **kwargs):
        try:
            action = pi_action
            log_prob = self.agent.get_log_prob(observation, action)
            advantage = self.to_tensor(advantage)
            
            # In the case where a baseline is used
            if isinstance(log_prob, Tuple):
                values, log_prob, mask = log_prob
                values = values.squeeze(-1)
                advantage = advantage.reshape(-1, 1).broadcast_to(values.size())
                value_loss = torch.mean(((advantage - values)*mask)**2)
                with torch.no_grad():
                    residual_advantage = advantage - values
                pg_loss = -torch.mean(torch.sum(residual_advantage*log_prob*mask, dim=1))
            else:
                advantages = advantage.flatten()
                values = torch.zeros_like(advantages)
                residual_advantage = torch.zeros_like(advantages)
                pg_loss = -torch.mean(log_prob.flatten()*advantages)
                value_loss = torch.zeros_like(pg_loss)
                
            advantages = advantage.flatten()
            self.accelerator.backward(pg_loss+value_loss)
            
            # Move to CPU to save memory
            advantages = advantages.detach().cpu()
            values_cpu = values.detach().cpu()
            residual_advantage_cpu = residual_advantage.detach().cpu()
            pg_loss_cpu = pg_loss.detach().cpu().item()
            value_loss_cpu = value_loss.detach().cpu().item()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM error in actor loss: {e}")
                self.clear_gpu_cache()
                # Return empty dict with zeros to prevent training failure
                return {
                    "pg.loss": 0.0, "values.loss": 0.0,
                    "values.mean": 0.0, "values.max": 0.0, "values.min": 0.0, "values.std": 0.0,
                    "advantages.mean": 0.0, "advantages.max": 0.0, "advantages.min": 0.0, "advantages.std": 0.0,
                    "residual_advantages.mean": 0.0, "residual_advantages.max": 0.0,
                    "residual_advantages.min": 0.0, "residual_advantages.std": 0.0,
                }
            else:
                raise e
                
        # Clear intermediate tensors
        del log_prob, advantage, values, residual_advantage, pg_loss, value_loss
        self.clear_gpu_cache()
        
        return {
            "pg.loss": pg_loss_cpu,
            "values.loss": value_loss_cpu,
            "values.mean": values_cpu.mean(),
            "values.max": torch.max(values_cpu),
            "values.min": torch.min(values_cpu),
            "values.std": torch.std(values_cpu),
            "advantages.mean": advantages.mean(),
            "advantages.max": torch.max(advantages),
            "advantages.min": torch.min(advantages),
            "advantages.std": torch.std(advantages),
            "residual_advantages.mean": residual_advantage_cpu.mean(),
            "residual_advantages.max": torch.max(residual_advantage_cpu),
            "residual_advantages.min": torch.min(residual_advantage_cpu),
            "residual_advantages.std": torch.std(residual_advantage_cpu),
        }

    def update(self, replay_buffer, no_update_actor=False):
        """Memory-optimized update method with batching and error handling."""
        self.step += 1
        info = {}
        critic_info_list = []
        
        # Clear GPU cache before starting
        self.clear_gpu_cache()
        
        # --- Critic Update ---
        batch_size = min(replay_buffer.batch_size, self.max_micro_batch)
        print(f"Using batch size {batch_size} for training")
        
        try:
            # Pre-sample data and ensure it's on the GPU
            data = []
            for _ in range(self.grad_accum_steps):
                try:
                    batch_data = [replay_buffer.sample(1) for _ in range(batch_size)]
                    for d in batch_data:
                        for k, v in d.items():
                            d[k] = v[0]
                    data.extend(batch_data)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM when sampling data, trying with half batch size")
                        if batch_size > 1:
                            half_batch = batch_size // 2
                            for _ in range(2):  # Do two half-batches instead
                                batch_data = [replay_buffer.sample(1) for _ in range(half_batch)]
                                for d in batch_data:
                                    for k, v in d.items():
                                        d[k] = v[0]
                                data.extend(batch_data)
                                self.clear_gpu_cache()
                    else:
                        raise e
                        
            # Increase num_workers for faster data loading but keep prefetch factor low to save memory
            dataloader = DataLoader(
                DummyDataset(data), 
                batch_size=batch_size,
                num_workers=2,  # Reduced from 4 
                shuffle=True, 
                pin_memory=True,
                prefetch_factor=2  # Limit prefetching to save memory
            )
            dataloader = self.accelerator.prepare(dataloader)
            
            # Training critic for self.epochs
            for epoch in tqdm(range(self.epochs), desc="Critic Epochs", leave=True):
                self.critic_optimizer.zero_grad()
                batches_processed = 0
                
                for batch in tqdm(dataloader, desc=f"Critic Update (Epoch {epoch+1}/{self.epochs})", leave=False):
                    try:
                        critic_info = self.critic_loss(**batch)
                        critic_info_list.append(critic_info)
                        batches_processed += 1
                        
                        # Periodically clear cache
                        if batches_processed % 4 == 0:
                            self.clear_gpu_cache()
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"OOM during critic update, skipping batch")
                            self.clear_gpu_cache()
                        else:
                            raise e
                
                self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # Soft-update target critic after critic updates
                self.agent.soft_update_target_critic(tau=self.tau)
                
                # Clear cache after each epoch
                self.clear_gpu_cache()
            
            if critic_info_list:
                info.update(dict_mean(critic_info_list))
            
        except Exception as e:
            print(f"Error during critic update: {e}")
            # Continue with actor update even if critic update fails
        
        # --- Actor Update ---
        if not no_update_actor:
            print(">>> Updating actor")
            actor_info_list = []
            
            try:
                # Clear cache before actor update
                self.clear_gpu_cache()
                
                # Pre-sample data for the actor update with memory optimization
                data = []
                for _ in range(self.grad_accum_steps):
                    try:
                        batch_data = [replay_buffer.sample(1) for _ in range(batch_size)]
                        for d in batch_data:
                            for k, v in d.items():
                                d[k] = v[0]
                        data.extend(batch_data)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"OOM when sampling data for actor, trying with half batch size")
                            if batch_size > 1:
                                half_batch = batch_size // 2
                                for _ in range(2):  # Do two half-batches instead
                                    batch_data = [replay_buffer.sample(1) for _ in range(half_batch)]
                                    for d in batch_data:
                                        for k, v in d.items():
                                            d[k] = v[0]
                                    data.extend(batch_data)
                                    self.clear_gpu_cache()
                        else:
                            raise e
                
                dataloader = DataLoader(
                    DummyDataset(data), 
                    batch_size=batch_size, 
                    num_workers=2,  # Reduced from 4
                    shuffle=False, 
                    pin_memory=True,
                    prefetch_factor=2  # Limit prefetching
                )
                dataloader = self.accelerator.prepare(dataloader)
                
                # Training actor for self.actor_epochs
                for epoch in tqdm(range(self.actor_epochs), desc="Actor Epochs", leave=True):
                    self.lm_optimizer.zero_grad()
                    batches_processed = 0
                    
                    for batch in tqdm(dataloader, desc=f"Actor Update (Epoch {epoch+1}/{self.actor_epochs})", leave=False):
                        try:
                            # Calculate advantages in a no_grad block to save memory
                            with torch.no_grad():
                                pi_action = self.agent.get_action(batch["observation"])
                                q1, q2, v1, v2 = self.agent.critic(batch["observation"], pi_action)
                                q = torch.minimum(q1, q2)
                                v = torch.minimum(v1, v2)
                                advantages = q - v
                            
                            actor_info = self.actor_loss(**batch, pi_action=pi_action, advantage=advantages)
                            actor_info_list.append(actor_info)
                            batches_processed += 1
                            
                            # Clean up intermediate tensors
                            del pi_action, q1, q2, v1, v2, q, v, advantages
                            
                            # Periodically clear cache
                            if batches_processed % 4 == 0:
                                self.clear_gpu_cache()
                                
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                print(f"OOM during actor update, skipping batch")
                                self.clear_gpu_cache()
                            else:
                                raise e
                    
                    self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.lm_optimizer.step()
                    
                    # Clear cache after each epoch
                    self.clear_gpu_cache()
                
                if actor_info_list:
                    info.update(dict_mean(actor_info_list))
                
            except Exception as e:
                print(f"Error during actor update: {e}")
            
        # Final cache clear
        self.clear_gpu_cache()
        return info

    def save(self, path):
        """Memory-optimized save method."""
        # Clear cache before saving
        self.clear_gpu_cache()
        
        try:
            # Save all at once
            torch.save({
                'model_state_dict': self.accelerator.unwrap_model(self.agent.model).state_dict(),
                'critic_state_dict': self.accelerator.unwrap_model(self.agent.critic).state_dict(),
                'target_critic_state_dict': self.accelerator.unwrap_model(self.agent.target_critic).state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'lm_optimizer_state_dict': self.lm_optimizer.state_dict()
            }, path)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM during save, trying sequential saves")
                # Save model state dict first
                torch.save({
                    'model_state_dict': self.accelerator.unwrap_model(self.agent.model).state_dict(),
                }, path + '.model')
                self.clear_gpu_cache()
                
                # Save critic state dict
                torch.save({
                    'critic_state_dict': self.accelerator.unwrap_model(self.agent.critic).state_dict(),
                }, path + '.critic')
                self.clear_gpu_cache()
                
                # Save target critic state dict
                torch.save({
                    'target_critic_state_dict': self.accelerator.unwrap_model(self.agent.target_critic).state_dict(),
                }, path + '.target_critic')
                self.clear_gpu_cache()
                
                # Save optimizer state dicts
                torch.save({
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'lm_optimizer_state_dict': self.lm_optimizer.state_dict()
                }, path + '.optimizers')
                
                print(f"Saved model components separately to {path}.*")
            else:
                raise e
        
        # Clear cache after saving
        self.clear_gpu_cache()

    def load(self, path):
        """Memory-optimized load method."""
        # Clear cache before loading
        self.clear_gpu_cache()
        
        try:
            # Try loading all at once
            checkpoint = torch.load(path, map_location=self.accelerator.device)
            self.agent.model.load_state_dict(checkpoint['model_state_dict'])
            self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.agent.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.lm_optimizer.load_state_dict(checkpoint['lm_optimizer_state_dict'])
        except (RuntimeError, KeyError) as e:
            if "out of memory" in str(e) or isinstance(e, KeyError):
                print(f"Error during load: {e}, trying to load components separately")
                
                # Check if separate files exist
                if os.path.exists(path + '.model'):
                    # Load model state dict
                    checkpoint = torch.load(path + '.model', map_location=self.accelerator.device)
                    self.agent.model.load_state_dict(checkpoint['model_state_dict'])
                    self.clear_gpu_cache()
                    
                    # Load critic state dict
                    checkpoint = torch.load(path + '.critic', map_location=self.accelerator.device)
                    self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
                    self.clear_gpu_cache()
                    
                    # Load target critic state dict
                    checkpoint = torch.load(path + '.target_critic', map_location=self.accelerator.device)
                    self.agent.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
                    self.clear_gpu_cache()
                    
                    # Load optimizer state dicts
                    checkpoint = torch.load(path + '.optimizers', map_location=self.accelerator.device)
                    self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                    self.lm_optimizer.load_state_dict(checkpoint['lm_optimizer_state_dict'])
                    
                    print(f"Loaded model components separately from {path}.*")
                else:
                    raise FileNotFoundError(f"Could not find model components at {path}.*")
            else:
                raise e
        
        # Clear cache after loading
        self.clear_gpu_cache()
        return self.agent