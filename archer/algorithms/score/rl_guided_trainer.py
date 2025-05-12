import torch
import copy
import transformers
import os
from tqdm import tqdm
import torch.nn.functional as F
from archer.algorithms.score.trainer import SCoReTrainer
from archer.prompts import generate_smart_correction_prompt, format_math_self_correction_prompt

class RLGuidedSCoReTrainer(SCoReTrainer):
    """
    Extended SCoRe trainer that also trains the guidance model using REINFORCE.
    The guidance model learns to provide better correction hints by receiving
    rewards based on whether the base model gets the correct answer after
    following the guidance.
    
    Memory-optimized implementation to handle larger models.
    """
    def __init__(
        self,
        agent,
        tokenizer,
        accelerator,
        guidance_model=None,
        guidance_lr: float = 1e-6,
        guidance_kl_coef: float = 0.05,
        train_guidance_model: bool = True,
        max_micro_batch: int = 4,  # Reduced from 8 to 4 for better memory efficiency
        use_gradient_checkpointing: bool = True,
        use_memory_efficient_attention: bool = True,
        **kwargs
    ):
        super().__init__(
            agent=agent, 
            tokenizer=tokenizer, 
            accelerator=accelerator,
            max_micro_batch=max_micro_batch,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_memory_efficient_attention=use_memory_efficient_attention,
            **kwargs
        )
        self.train_guidance_model = train_guidance_model
        self.guidance_kl_coef = guidance_kl_coef

        # Initialize or clone guidance model with memory optimizations
        if guidance_model is not None:
            self.guidance_model = guidance_model
        else:
            try:
                from peft import PeftModel
                if isinstance(self.ref_model, PeftModel):
                    print("Cloning PeftModel with memory optimizations...")
                    # Move ref_model to CPU before copying to save memory
                    self.ref_model.to('cpu')
                    import copy as _copy
                    self.guidance_model = _copy.deepcopy(self.ref_model)
                    # Move models back to appropriate devices
                    self.ref_model.to(agent.model.device)
                    self.guidance_model.to(agent.model.device)
                    torch.cuda.empty_cache()
                else:
                    print("Initializing new model with memory optimizations...")
                    model_cls = type(self.ref_model)
                    cfg_dict = {k: v for k, v in self.ref_model.config.to_dict().items() if not k.startswith('_')}
                    
                    # Add memory optimization configs
                    cfg_dict['low_cpu_mem_usage'] = True
                    if use_memory_efficient_attention:
                        cfg_dict['use_memory_efficient_attention'] = True
                        cfg_dict['use_flash_attention'] = True
                    
                    # Initialize model on CPU first
                    self.guidance_model = model_cls(**cfg_dict).to('cpu')
                    
                    # Load state dict in chunks to save memory
                    chunk_size = 100  # Process state dict in chunks
                    ref_sd = {}
                    keys = list(self.ref_model.state_dict().keys())
                    for i in range(0, len(keys), chunk_size):
                        chunk_keys = keys[i:i + chunk_size]
                        for k in chunk_keys:
                            ref_sd[k] = self.ref_model.state_dict()[k].to('cpu')
                    
                    self.guidance_model.load_state_dict(ref_sd)
                    del ref_sd
                    # Move to target device
                    self.guidance_model.to(agent.model.device)
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error in optimized model initialization: {e}")
                print("Falling back to basic initialization...")
                if isinstance(self.ref_model, PeftModel):
                    self.guidance_model = _copy.deepcopy(self.ref_model).to(agent.model.device)
                else:
                    model_cls = type(self.ref_model)
                    self.guidance_model = model_cls(self.ref_model.config).to(agent.model.device)
                    self.guidance_model.load_state_dict(self.ref_model.state_dict())
                torch.cuda.empty_cache()

        # Enable memory optimizations for guidance model
        if use_gradient_checkpointing and hasattr(self.guidance_model, "gradient_checkpointing_enable"):
            self.guidance_model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for guidance model")
            
        if use_memory_efficient_attention:
            if hasattr(self.guidance_model, "config"):
                if hasattr(self.guidance_model.config, "use_memory_efficient_attention"):
                    self.guidance_model.config.use_memory_efficient_attention = True
                    print("Memory efficient attention enabled for guidance model")
                if hasattr(self.guidance_model.config, "use_flash_attention"):
                    self.guidance_model.config.use_flash_attention = True
                    print("Flash attention enabled for guidance model")

        # Guidance optimizer and snapshot parameters with memory optimization
        if self.train_guidance_model:
            # Initialize optimizer with gradient accumulation
            self.guidance_optimizer = torch.optim.Adam(
                self.guidance_model.parameters(),
                lr=guidance_lr
            )
            self.guidance_optimizer = self.accelerator.prepare(self.guidance_optimizer)
            
            # Store reference parameters on CPU to save GPU memory
            self.ref_params = {}
            for name, param in self.guidance_model.named_parameters():
                if param.requires_grad:
                    self.ref_params[name] = param.detach().cpu().clone()
            
            print(f"Guidance model training enabled with lr={guidance_lr}")
        else:
            print("Guidance model training disabled")

    def guidance_get_log_prob(self, prompts, actions):
        """
        Compute log-probs of `actions` under self.guidance_model by swapping it into agent.
        Returns a Tensor of shape (batch,).
        Memory-optimized implementation.
        """
        orig = self.agent.model
        self.agent.model = self.guidance_model
        try:
            # Process in smaller chunks to save memory
            batch_size = len(prompts)
            max_chunk_size = min(self.max_micro_batch, 4)  # Use even smaller chunks
            log_probs_list = []
            
            for i in range(0, batch_size, max_chunk_size):
                # Clear cache at start of each chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                end_idx = min(i + max_chunk_size, batch_size)
                chunk_prompts = prompts[i:end_idx]
                chunk_actions = actions[i:end_idx]
                
                try:
                    chunk_log_probs = self.agent.get_log_prob(chunk_prompts, chunk_actions)
                    log_probs_list.append(chunk_log_probs)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM in chunk {i}-{end_idx}, trying with half size")
                        # Try again with half the chunk size
                        half_size = (end_idx - i) // 2
                        if half_size > 0:
                            # Process first half
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            mid_idx = i + half_size
                            chunk_log_probs = self.agent.get_log_prob(
                                prompts[i:mid_idx],
                                actions[i:mid_idx]
                            )
                            log_probs_list.append(chunk_log_probs)
                            
                            # Process second half
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            chunk_log_probs = self.agent.get_log_prob(
                                prompts[mid_idx:end_idx],
                                actions[mid_idx:end_idx]
                            )
                            log_probs_list.append(chunk_log_probs)
                    else:
                        raise e
                
                # Clean up to save memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Combine results
            log_probs = torch.cat(log_probs_list, dim=0)
            
        finally:
            self.agent.model = orig
            
        return log_probs

    def generate_custom_guidance(self, problems, initial_solutions, batch_size: int = 4):  # Reduced batch size
        """
        Generate tailored guidance hints for each (problem, initial_solution) pair.
        Memory-optimized implementation with careful cache management.
        """
        # Safe to clear before starting new generation
        self.clear_gpu_cache()
        
        # ---------- normalise inputs ----------
        is_batch = isinstance(problems, list)
        if not is_batch:
            problems, initial_solutions = [problems], [initial_solutions]

        device = self.agent.model.device
        analysis_prompts = []
        guidance_texts = []

        # ---------- tokenizer housekeeping ----------
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            for start in tqdm(
                range(0, len(problems), batch_size),
                desc="Generating guidance"
            ):
                # Clear cache at start of each batch
                self.clear_gpu_cache()
                
                # ----- slice current batch -----
                end = min(start + batch_size, len(problems))
                batch_probs = problems[start:end]
                batch_sols = initial_solutions[start:end]

                try:
                    # ----- build analysis prompts -----
                    batch_prompts = [
                        (
                            "You are an expert math tutor reviewing a student's solution to a math problem.\n\n"
                            f"PROBLEM:{p}\n\n"
                            f"INITIAL SOLUTION:{s}\n\n"
                            "PROMPT: First, analyze the solution for errors or misconceptions. "
                            "Then, write a brief, helpful instruction that will guide the student toward "
                            "correcting their solution. Your instruction should be specific to the errors you "
                            "identified, but don't solve the problem for them. Your response should be ONLY the "
                            "instruction for the student to improve their solution, nothing else. "
                            "DO NOT include ANY SOLUTION.\n\n"
                            "GUIDING INSTRUCTION:"
                        )
                        for p, s in zip(batch_probs, batch_sols)
                    ]
                    analysis_prompts.extend(batch_prompts)

                    # ----- tokenise with memory optimization -----
                    inputs = self.tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024,
                    ).to(device)

                    # ----- generate guidance text with memory optimization -----
                    ctx = torch.enable_grad() if getattr(self, "train_guidance_model", False) else torch.no_grad()
                    with ctx:
                        outputs = self.guidance_model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=256,
                            temperature=0.7,
                            top_p=0.95,
                            do_sample=True,
                            num_beams=2,
                            use_cache=not getattr(self, "use_gradient_checkpointing", False),  # Disable KV cache if using gradient checkpointing
                        )

                        # ----- decode JUST the generated portion (strip input) -----
                        input_lens = [len(ids) for ids in inputs["input_ids"]]
                        for j, seq in enumerate(outputs):
                            instr = self.tokenizer.decode(
                                seq[input_lens[j]:], skip_special_tokens=True
                            ).strip()
                            guidance_texts.append(instr)

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM in batch {start}-{end}, trying with half size")
                        # Try again with half the batch size
                        half_size = (end - start) // 2
                        if half_size > 0:
                            # Process first half
                            self.clear_gpu_cache()
                            mid = start + half_size
                            self._process_guidance_batch(
                                problems[start:mid],
                                initial_solutions[start:mid],
                                analysis_prompts,
                                guidance_texts
                            )
                            
                            # Process second half
                            self.clear_gpu_cache()
                            self._process_guidance_batch(
                                problems[mid:end],
                                initial_solutions[mid:end],
                                analysis_prompts,
                                guidance_texts
                            )
                    else:
                        raise e

                # Only clear after we've processed the outputs
                del inputs, outputs
                self.clear_gpu_cache()

        except Exception as e:
            # any failure → gracefully fall back to static template generation
            print(f"[generate_custom_guidance] Falling back due to error ⇒ {e}")
            fallback_prompts = generate_smart_correction_prompt(
                problems, initial_solutions,
                correction_model=None  # static template path
            )
            # `generate_smart_correction_prompt` returns list when given lists
            if not is_batch:
                return [fallback_prompts], fallback_prompts
            return fallback_prompts, ["" for _ in fallback_prompts]

        # ---------- shape of return matches shape of input ----------
        if not is_batch:
            return analysis_prompts[0], guidance_texts[0]
            
        # Safe to clear before returning
        self.clear_gpu_cache()
        return analysis_prompts, guidance_texts

    def _process_guidance_batch(self, problems, solutions, analysis_prompts, guidance_texts):
        """Helper method to process a single guidance generation batch."""
        device = self.agent.model.device
        
        # Build prompts
        batch_prompts = [
            (
                "You are an expert math tutor reviewing a student's solution to a math problem.\n\n"
                f"PROBLEM:{p}\n\n"
                f"INITIAL SOLUTION:{s}\n\n"
                "PROMPT: First, analyze the solution for errors or misconceptions. "
                "Then, write a brief, helpful instruction that will guide the student toward "
                "correcting their solution. Your instruction should be specific to the errors you "
                "identified, but don't solve the problem for them. Your response should be ONLY the "
                "instruction for the student to improve their solution, nothing else. "
                "DO NOT include ANY SOLUTION.\n\n"
                "GUIDING INSTRUCTION:"
            )
            for p, s in zip(problems, solutions)
        ]
        analysis_prompts.extend(batch_prompts)
        
        # Tokenize
        inputs = self.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)
        
        # Generate
        ctx = torch.enable_grad() if getattr(self, "train_guidance_model", False) else torch.no_grad()
        with ctx:
            outputs = self.guidance_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                num_beams=2,
                use_cache=not getattr(self, "use_gradient_checkpointing", False),
            )
            
            # Decode
            input_lens = [len(ids) for ids in inputs["input_ids"]]
            for j, seq in enumerate(outputs):
                instr = self.tokenizer.decode(
                    seq[input_lens[j]:], skip_special_tokens=True
                ).strip()
                guidance_texts.append(instr)
        
        # Clean up
        del inputs, outputs
        self.clear_gpu_cache()

    def calculate_guidance_log_probs(self, analysis_prompts, guidance_texts):
        """
        Given the analysis_prompts and guidance_texts, compute per-example log-probs.
        Memory-optimized implementation.
        """
        return self.guidance_get_log_prob(analysis_prompts, guidance_texts)

    def train_guidance(self, trajectories):
        """
        Train the guidance model using REINFORCE.
        Memory-optimized implementation with regular cache clearing.
        """
        if not self.train_guidance_model:
            return {}

        # Clear cache before starting
        self.clear_gpu_cache()

        # Extract data
        problems = [t[0]['observation'] for t in trajectories]
        sols = [t[0]['action_turn1'] for t in trajectories]
        raw_r = [t[0]['reward_turn2'] for t in trajectories]

        # Generate prompts and hints in smaller batches to save memory
        batch_size = len(problems)
        max_guidance_batch = min(self.max_micro_batch, 4)  # Use even smaller batches
        
        analysis_prompts = []
        guidance_texts = []
        
        for i in range(0, batch_size, max_guidance_batch):
            # Clear cache at start of each batch
            self.clear_gpu_cache()
            
            end_idx = min(i + max_guidance_batch, batch_size)
            
            try:
                batch_analysis, batch_guidance = self.generate_custom_guidance(
                    problems[i:end_idx], 
                    sols[i:end_idx]
                )
                
                analysis_prompts.extend(batch_analysis)
                guidance_texts.extend(batch_guidance)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM in batch {i}-{end_idx}, trying with half size")
                    # Try again with half the batch size
                    half_size = (end_idx - i) // 2
                    if half_size > 0:
                        # Process first half
                        self.clear_gpu_cache()
                        mid_idx = i + half_size
                        batch_analysis, batch_guidance = self.generate_custom_guidance(
                            problems[i:mid_idx], 
                            sols[i:mid_idx]
                        )
                        analysis_prompts.extend(batch_analysis)
                        guidance_texts.extend(batch_guidance)
                        
                        # Process second half
                        self.clear_gpu_cache()
                        batch_analysis, batch_guidance = self.generate_custom_guidance(
                            problems[mid_idx:end_idx], 
                            sols[mid_idx:end_idx]
                        )
                        analysis_prompts.extend(batch_analysis)
                        guidance_texts.extend(batch_guidance)
                else:
                    raise e
            
            # Clean up to save memory
            self.clear_gpu_cache()
                
        if not guidance_texts:
            return {'guidance_loss': 0.0}

        device = self.agent.model.device
        # Symmetric rewards: {0->-1,1->+1}
        shaped_r = [1 if r>0 else -1 for r in raw_r]
        rewards_tensor = torch.tensor(shaped_r, dtype=torch.float, device=device)

        # Process in smaller batches for memory efficiency
        total_loss = 0.0
        total_samples = 0
        
        self.guidance_optimizer.zero_grad()
        
        for i in range(0, batch_size, max_guidance_batch):
            # Clear cache at start of each batch
            self.clear_gpu_cache()
            
            end_idx = min(i + max_guidance_batch, batch_size)
            current_batch_size = end_idx - i
            
            try:
                # Get batch data
                batch_prompts = analysis_prompts[i:end_idx]
                batch_texts = guidance_texts[i:end_idx]
                batch_rewards = rewards_tensor[i:end_idx]
                
                # Calculate log probabilities for this batch
                self.guidance_model.train()  # Ensure model is in training mode
                logp_tensor = self.calculate_guidance_log_probs(batch_prompts, batch_texts)
                
                # Calculate policy gradient loss for this batch
                batch_policy_loss = -torch.mean(logp_tensor * batch_rewards)
                
                # Scale loss by batch size and accumulate gradients
                scaled_loss = batch_policy_loss * current_batch_size / max_guidance_batch
                self.accelerator.backward(scaled_loss)
                
                total_loss += batch_policy_loss.item() * current_batch_size
                total_samples += current_batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM in batch {i}-{end_idx}, trying with half size")
                    # Try again with half the batch size
                    half_size = (end_idx - i) // 2
                    if half_size > 0:
                        # Process first half
                        self.clear_gpu_cache()
                        mid_idx = i + half_size
                        self._process_policy_gradient_batch(
                            analysis_prompts[i:mid_idx],
                            guidance_texts[i:mid_idx],
                            rewards_tensor[i:mid_idx],
                            half_size,
                            max_guidance_batch,
                            total_loss,
                            total_samples
                        )
                        
                        # Process second half
                        self.clear_gpu_cache()
                        self._process_policy_gradient_batch(
                            analysis_prompts[mid_idx:end_idx],
                            guidance_texts[mid_idx:end_idx],
                            rewards_tensor[mid_idx:end_idx],
                            half_size,
                            max_guidance_batch,
                            total_loss,
                            total_samples
                        )
                else:
                    raise e
            
            # Clean up to save memory
            del batch_prompts, batch_texts, batch_rewards
            if 'logp_tensor' in locals(): del logp_tensor
            if 'batch_policy_loss' in locals(): del batch_policy_loss
            if 'scaled_loss' in locals(): del scaled_loss
            self.clear_gpu_cache()
        
        # Update the guidance model after processing all batches
        self.accelerator.clip_grad_norm_(self.guidance_model.parameters(), self.max_grad_norm)
        self.guidance_optimizer.step()
        
        # Calculate average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        # Final cache clear
        self.clear_gpu_cache()

        return {
            'guidance_loss': avg_loss,
            'guidance_reward_mean': rewards_tensor.mean().item(),
        }

    def _process_policy_gradient_batch(
        self,
        batch_prompts,
        batch_texts,
        batch_rewards,
        batch_size,
        max_batch_size,
        total_loss,
        total_samples
    ):
        """Helper method to process a single policy gradient batch with memory optimizations."""
        # Calculate log probabilities
        self.guidance_model.train()
        logp_tensor = self.calculate_guidance_log_probs(batch_prompts, batch_texts)
        
        # Calculate and apply loss
        batch_policy_loss = -torch.mean(logp_tensor * batch_rewards)
        scaled_loss = batch_policy_loss * batch_size / max_batch_size
        self.accelerator.backward(scaled_loss)
        
        # Update totals
        total_loss += batch_policy_loss.item() * batch_size
        total_samples += batch_size
        
        # Clean up
        del logp_tensor, batch_policy_loss, scaled_loss
        self.clear_gpu_cache()
        
        return total_loss, total_samples

    def update(self, trajectories, no_update_actor=False):
        """
        Update both the base model and guidance model.
        Memory-optimized implementation.
        """
        # Clear cache before starting
        self.clear_gpu_cache()
        
        info = super().update(trajectories, no_update_actor)
        if self.train_guidance_model and not no_update_actor:
            guidance_info = self.train_guidance(trajectories)
            info.update(guidance_info)
            
        # Final cache clear
        self.clear_gpu_cache()
        return info

    def save(self, path):
        """
        Save the models with memory optimization.
        """
        # Clear CUDA cache before saving
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        try:
            # Save base model
            super().save(path)
            
            # Save guidance model
            gpath = path.replace('.pt', '_guidance.pt')
            torch.save({
                'model_state_dict': self.guidance_model.state_dict(),
                'optimizer_state_dict': getattr(self, 'guidance_optimizer', None).state_dict() if hasattr(self, 'guidance_optimizer') else None
            }, gpath)
            print(f"Saved guidance model to {gpath}")
        except RuntimeError as e:
            print(f"Warning: Memory issue when saving. Error: {e}")
            print("Trying simplified save...")
            
            # Try a simplified save approach to avoid OOM
            if hasattr(self, 'guidance_model'):
                gpath = path.replace('.pt', '_guidance.pt')
                # Save state dict directly without optimizer state
                torch.save({
                    'model_state_dict': self.guidance_model.state_dict(),
                }, gpath)
                print(f"Saved guidance model only (no optimizer state) to {gpath}")

    def load(self, path):
        """
        Load the models with memory optimization.
        """
        # Clear CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Load base model
        agent = super().load(path)
        
        # Load guidance model
        gpath = path.replace('.pt', '_guidance.pt')
        if os.path.exists(gpath):
            try:
                ckpt = torch.load(gpath)
                self.guidance_model.load_state_dict(ckpt['model_state_dict'])
                if hasattr(self, 'guidance_optimizer') and 'optimizer_state_dict' in ckpt:
                    self.guidance_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print(f"Loaded guidance model from {gpath}")
            except RuntimeError as e:
                print(f"Warning: Error loading guidance model: {e}")
                print("Continuing without loading guidance model optimizer state")
                
        return agent
