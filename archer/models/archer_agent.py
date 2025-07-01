import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, List, Dict, Any
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from archer.models.critic import DoubleCritic


class ArcherAgent(torch.nn.Module):
    def __init__(self, device, accelerator, policy_lm = "llama3.2", critic_lm = "roberta-base", 
                cache_dir = '~/.cache', dropout = 0.5, TEMPLATE = None, use_lora=False,
                do_sample = True, temperature = 0.9, max_new_tokens = 512, use_bfloat16 = False, eos_str = '\n', use_gradient_checkpointing = False, use_memory_efficient_attention = False,
                load_in_8bit = False):
        super(ArcherAgent, self).__init__()
        
        # Memory optimization: Clear cache at start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 1. Load base model with memory optimizations
        if True:
            # Use 8-bit quantization if requested to save memory
            if load_in_8bit:
                print("Loading model in 8-bit precision to save memory")
                self.model = AutoModelForCausalLM.from_pretrained(
                    policy_lm, 
                    cache_dir=cache_dir,
                    load_in_8bit=True,
                    device_map="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    policy_lm, 
                    cache_dir=cache_dir,
                    torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                    low_cpu_mem_usage=True
                ).to(device)
            
        # Enable gradient checkpointing to save memory if requested
        if use_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for model")
            
        # Enable memory efficient attention if requested
        if use_memory_efficient_attention and hasattr(self.model, "config"):
            if hasattr(self.model.config, "use_memory_efficient_attention"):
                self.model.config.use_memory_efficient_attention = True
                print("Memory efficient attention enabled for model")
            elif hasattr(self.model.config, "attention_implementation"):
                self.model.config.attention_implementation = "flash_attention_2"
                print("Flash Attention 2 enabled for model")
        
        # Clear cache after base model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if use_lora:
            from peft import LoraConfig, TaskType, get_peft_model
            lora_config = LoraConfig(
                r=16,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=32,
                lora_dropout=0.05
            )
            
            # Save memory by clearing CUDA cache before creating PEFT model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Use CPU for initial PEFT model creation to save GPU memory
            original_device = self.model.device
            self.model = self.model.cpu()
            
            try:
                self.model = get_peft_model(self.model, lora_config)
                print("Using LoRA")
                self.model.print_trainable_parameters()
                
                # Only make LoRA parameters trainable
                for param in self.model.parameters():
                    param.requires_grad = False
                for name, param in self.model.named_parameters():
                    if "lora" in name:
                        param.requires_grad = True
                        
                # Move back to original device
                self.model = self.model.to(original_device)
            except Exception as e:
                print(f"Error creating PEFT model: {e}")
                print("Falling back to full-parameter fine-tuning")
                # Move back to original device in case of error
                self.model = self.model.to(original_device)

        self.template = TEMPLATE
        self.policy_lm = policy_lm
        
        # 2. Initialize critic on GPU
        print("Initializing main critic...")
        self.critic = DoubleCritic(device, accelerator, critic_lm = critic_lm, cache_dir = cache_dir, in_dim = 768, out_dim = 1)
        
        # Clear cache after critic initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 3. CRITICAL: Initialize target critic on CPU to save GPU memory
        print("Initializing target critic on CPU to save memory...")
        cpu_device = torch.device('cpu')
        self.target_critic = DoubleCritic(cpu_device, accelerator, critic_lm = critic_lm, cache_dir = cache_dir, in_dim = 768, out_dim = 1)
        
        # 4. MEMORY-EFFICIENT TARGET UPDATE: Copy parameters without full GPU allocation
        print("Performing memory-efficient target critic initialization...")
        self._memory_efficient_target_update(tau=1.0)
        
        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True, cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'left'
        if self.tokenizer.pad_token is None:
            special_tokens_dict = {'pad_token': '[PAD]'}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax(dim= -1)
        self.do_sample = do_sample
        self.temperature = temperature
        self.accelerator = accelerator
        self.max_new_tokens = max_new_tokens
        self.eos_str = eos_str

    def prepare(self):
        self.model, self.critic, self.target_critic = self.accelerator.prepare(
            self.model, self.critic, self.target_critic
        )

    def get_action(self, observation):
        # Debug print for observation info
        
        if self.template is not None:
            if isinstance(self.template, list):
                prompts = self.template
            else:
                prompts = observation
        else:
            from archer.prompts.math import format_math_prompt
            prompts = [format_math_prompt(example) for example in observation]
        
        # Process input prompts in smaller batches to avoid CUDA OOM errors
        batch_size = len(prompts)
        max_safe_batch = 4  # Start with a small safe batch size
        all_actions = []
        
        # Clear CUDA cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for i in range(0, batch_size, max_safe_batch):
            end_idx = min(i + max_safe_batch, batch_size)
            batch_prompts = prompts[i:end_idx]
            
            try:
                # Process this batch
                inputs = self.tokenizer(
                    batch_prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=1024,
                    return_attention_mask=True
                ).to(self.model.device)
                
                with torch.no_grad():
                    # Try to use bfloat16 if available to save memory
                    if hasattr(torch, 'bfloat16') and torch.cuda.is_available():
                        inputs = {k: v.to(torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point() else v 
                                for k, v in inputs.items()}
                    
                    # Use more memory-efficient generation settings
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=self.do_sample,
                        temperature=self.temperature,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        repetition_penalty=1.0,
                    )
                
                # Decode outputs and add to the result list
                batch_actions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                batch_clean_actions = [action.split("I hope it is correct.")[-1] for action in batch_actions]
                all_actions.extend(batch_clean_actions)
                
                # Clear memory after processing each batch
                del inputs, outputs, batch_actions, batch_clean_actions
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    # If OOM, try with an even smaller batch size
                    print(f"CUDA OOM with batch size {max_safe_batch}, trying with smaller batches...")
                    
                    # Process one prompt at a time as a fallback
                    for j in range(i, end_idx):
                        single_prompt = [prompts[j]]
                        
                        # Clear cache before each single prompt
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                        try:
                            inputs = self.tokenizer(
                                single_prompt, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True,
                                max_length=1024,
                                return_attention_mask=True
                            ).to(self.model.device)
                            
                            with torch.no_grad():
                                # Move to CPU if still out of memory
                                if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0.9 * torch.cuda.get_device_properties(0).total_memory:
                                    print("Memory usage too high, offloading model to CPU...")
                                    model_device = self.model.device
                                    # Temporarily move model to CPU
                                    self.model = self.model.cpu()
                                    inputs = {k: v.cpu() for k, v in inputs.items()}
                                    
                                    outputs = self.model.generate(
                                        **inputs,
                                        max_new_tokens=self.max_new_tokens,
                                        do_sample=self.do_sample,
                                        temperature=self.temperature,
                                        pad_token_id=self.tokenizer.eos_token_id,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        use_cache=True,
                                        repetition_penalty=1.0,
                                    )
                                    
                                    # Move model back to original device
                                    self.model = self.model.to(model_device)
                                else:
                                    outputs = self.model.generate(
                                        **inputs,
                                        max_new_tokens=self.max_new_tokens,
                                        do_sample=self.do_sample,
                                        temperature=self.temperature,
                                        pad_token_id=self.tokenizer.eos_token_id,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        use_cache=True,
                                        repetition_penalty=1.0,
                                    )
                            
                            # Get action for this single prompt
                            single_action = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                            clean_action = single_action.split("I hope it is correct.")[-1]
                            all_actions.append(clean_action)
                            
                            # Clear memory after each prompt
                            del inputs, outputs, single_action
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                        except RuntimeError as e2:
                            print(f"Critical error generating action for prompt {j}: {e2}")
                            # Add an empty string as a fallback action
                            all_actions.append("")
                else:
                    # Re-raise if not a memory error
                    raise e

        return all_actions

    def get_q(self, observation, action, detach_model=False):
        return self.critic.get_q(observation, action, detach_model = detach_model)

    def get_v(self, inputs, detach_model=False):
        return self.critic.get_v(inputs, detach_model = detach_model)
    
    def get_target_q(self, observation, action, detach_model=False):
        """
        Modified to handle CPU-based target critic.
        Temporarily moves data to CPU for target computation.
        """
        # Move inputs to CPU for target critic computation
        original_device = observation[0] if isinstance(observation, list) else None
        
        # Use target critic on CPU
        with torch.no_grad():
            q1, q2 = self.target_critic.get_q(observation, action, detach_model=True)
            
            # Move results back to original device if needed
            if hasattr(q1, 'to'):
                q1 = q1.to(self.device)
                q2 = q2.to(self.device)
                
        return q1, q2

    def get_log_prob(self, observation, action):
        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        action_ids = self.tokenizer(action, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        # action_embeds = self.model.get_input_embeddings()(action_ids["input_ids"]).detach()
        # obs_embeds = self.model.get_input_embeddings()(obs_ids["input_ids"]).detach()
        input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim = 1)
        # input_embeds = torch.cat([obs_embeds, action_embeds], dim = 1)
        attention_mask = torch.cat([obs_ids["attention_mask"], action_ids["attention_mask"]],\
                                dim = 1)
        outputs = self.model(input_ids=input_ids, attention_mask = attention_mask)
        values = None
        if isinstance(outputs, Tuple):
            values, outputs = outputs
        prediction_probs = self.softmax(outputs.logits)
        selected_prediction_probs = torch.take_along_dim(prediction_probs[:, obs_ids["attention_mask"].size(1)-1:-1],\
                                                 action_ids["input_ids"].unsqueeze(2), dim=2).squeeze(2)
        if values is not None:
            return values[:, obs_ids["attention_mask"].size(1)-1:-1], torch.log(selected_prediction_probs)*action_ids["attention_mask"], action_ids["attention_mask"]
        else:
            return torch.sum(torch.log(selected_prediction_probs)*action_ids["attention_mask"], dim = 1)

    def _memory_efficient_target_update(self, tau):
        """
        Memory-efficient target critic update that avoids OOM by:
        1. Processing parameters in chunks
        2. Using CPU operations when possible
        3. Clearing cache frequently
        """
        print(f"Updating target critic with tau={tau} (memory-efficient)")
        
        # Get parameter iterators
        target_params = list(self.target_critic.parameters())
        source_params = list(self.critic.parameters())
        
        # Process parameters in chunks to avoid memory spikes
        chunk_size = 10  # Process 10 parameters at a time
        
        for i in range(0, len(target_params), chunk_size):
            end_idx = min(i + chunk_size, len(target_params))
            
            # Clear cache before processing each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            for j in range(i, end_idx):
                target_param = target_params[j]
                source_param = source_params[j]
                
                # Move source param to CPU temporarily to save GPU memory
                source_data = source_param.data.cpu()
                
                # Update target parameter (which is already on CPU)
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + source_data * tau
                )
                
                # Clean up temporary tensor
                del source_data
            
            # Clear cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("Target critic update completed")

    def soft_update_target_critic(self, tau):
        """Use memory-efficient update method"""
        self._memory_efficient_target_update(tau)

    def update_template(self, new_template):
        """Update the template used by the agent for formatting prompts.
        This allows dynamically switching between zero-shot and correction prompts.
        """
        self.template = new_template
        return self