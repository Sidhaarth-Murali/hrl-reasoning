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
        max_micro_batch: int = 8,  # Maximum size of micro-batches for memory efficiency
        use_gradient_checkpointing: bool = True,  # Whether to use gradient checkpointing
        use_memory_efficient_attention: bool = True,  # Whether to use memory efficient attention
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
            from peft import PeftModel
            if isinstance(self.ref_model, PeftModel):
                import copy as _copy
                self.guidance_model = _copy.deepcopy(self.ref_model).to(agent.model.device)
                torch.cuda.empty_cache()
            else:
                model_cls = type(self.ref_model)
                cfg_dict = {k: v for k, v in self.ref_model.config.to_dict().items() if not k.startswith('_')}
                self.guidance_model = model_cls(**cfg_dict).to(agent.model.device)
                ref_sd = {k: v.to('cpu') for k, v in self.ref_model.state_dict().items()}
                self.guidance_model.load_state_dict(ref_sd)
                del ref_sd
                torch.cuda.empty_cache()

        # Enable memory optimizations for guidance model
        if use_gradient_checkpointing and hasattr(self.guidance_model, "gradient_checkpointing_enable"):
            self.guidance_model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for guidance model")
            
        if use_memory_efficient_attention and hasattr(self.guidance_model, "use_memory_efficient_attention"):
            self.guidance_model.use_memory_efficient_attention = True
            print("Memory efficient attention enabled for guidance model")

        # Guidance optimizer and snapshot parameters
        if self.train_guidance_model:
            self.guidance_optimizer = torch.optim.Adam(
                self.guidance_model.parameters(), lr=guidance_lr
            )
            self.guidance_optimizer = self.accelerator.prepare(self.guidance_optimizer)
            self.ref_params = {
                name: param.detach().cpu().clone()
                for name, param in self.guidance_model.named_parameters()
                if param.requires_grad
            }
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
            max_chunk_size = self.max_micro_batch
            log_probs_list = []
            
            for i in range(0, batch_size, max_chunk_size):
                end_idx = min(i + max_chunk_size, batch_size)
                chunk_prompts = prompts[i:end_idx]
                chunk_actions = actions[i:end_idx]
                
                chunk_log_probs = self.agent.get_log_prob(chunk_prompts, chunk_actions)
                log_probs_list.append(chunk_log_probs)
                
                # Clean up to save memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Combine results
            log_probs = torch.cat(log_probs_list, dim=0)
            
        finally:
            self.agent.model = orig
            
        return log_probs

    def generate_custom_guidance(self, problems, initial_solutions, batch_size: int = 8):
        """
        Generate tailored guidance hints for each (problem, initial_solution) pair.
        Memory-optimized implementation.

        Args:
            problems (Union[str, List[str]]):  Single problem or list of problems.
            initial_solutions (Union[str, List[str]]):  Single draft solution or list of solutions.
            batch_size (int, optional):  # items processed per forward‑pass (controls VRAM).  Default = 8.

        Returns:
            Tuple[List[str], List[str]]  →  (analysis_prompts, guidance_texts)  **or**
            Tuple[str, str]              →  when a single problem/solution is given.
        """
        import torch
        from tqdm import tqdm

        # ---------- normalise inputs ----------
        is_batch = isinstance(problems, list)
        if not is_batch:
            problems, initial_solutions = [problems], [initial_solutions]

        device = self.agent.model.device
        analysis_prompts = []  # full text fed to model
        guidance_texts = []  # model‑generated instructions only

        # ---------- tokenizer housekeeping ----------
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            for start in tqdm(
                range(0, len(problems), batch_size),
                desc="Generating guidance"
            ):
                # ----- slice current batch -----
                batch_probs = problems[start:start + batch_size]
                batch_sols = initial_solutions[start:start + batch_size]

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
                        use_cache=True,
                    )

                # ----- decode JUST the generated portion (strip input) -----
                input_lens = [len(ids) for ids in inputs["input_ids"]]
                for j, seq in enumerate(outputs):
                    instr = self.tokenizer.decode(
                        seq[input_lens[j]:], skip_special_tokens=True
                    ).strip()
                    guidance_texts.append(instr)

                # ----- tidy up GPU memory -----
                del inputs, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
        return analysis_prompts, guidance_texts

    def calculate_guidance_log_probs(self, analysis_prompts, guidance_texts):
        """
        Given the analysis_prompts and guidance_texts, compute per-example log-probs.
        Memory-optimized implementation.
        """
        return self.guidance_get_log_prob(analysis_prompts, guidance_texts)

    def train_guidance(self, trajectories):
        """
        Train the guidance model using REINFORCE.
        Memory-optimized implementation.
        """
        if not self.train_guidance_model:
            return {}

        # Extract data
        problems = [t[0]['observation'] for t in trajectories]
        sols = [t[0]['action_turn1'] for t in trajectories]
        raw_r = [t[0]['reward_turn2'] for t in trajectories]

        # Generate prompts and hints in smaller batches to save memory
        batch_size = len(problems)
        max_guidance_batch = self.max_micro_batch
        
        analysis_prompts = []
        guidance_texts = []
        
        for i in range(0, batch_size, max_guidance_batch):
            end_idx = min(i + max_guidance_batch, batch_size)
            batch_slice = slice(i, end_idx)
            
            batch_analysis, batch_guidance = self.generate_custom_guidance(
                problems[batch_slice], 
                sols[batch_slice]
            )
            
            analysis_prompts.extend(batch_analysis)
            guidance_texts.extend(batch_guidance)
            
            # Clean up to save memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
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
            end_idx = min(i + max_guidance_batch, batch_size)
            batch_slice = slice(i, end_idx)
            current_batch_size = end_idx - i
            
            # Get batch data
            batch_prompts = analysis_prompts[i:end_idx]
            batch_texts = guidance_texts[i:end_idx]
            batch_rewards = rewards_tensor[batch_slice]
            
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
            
            # Clean up to save memory
            del batch_prompts, batch_texts, batch_rewards, logp_tensor, batch_policy_loss, scaled_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Update the guidance model after processing all batches
        self.accelerator.clip_grad_norm_(self.guidance_model.parameters(), self.max_grad_norm)
        self.guidance_optimizer.step()
        
        # Calculate average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        return {
            'guidance_loss': avg_loss,
            'guidance_reward_mean': rewards_tensor.mean().item(),
        }

    def update(self, trajectories, no_update_actor=False):
        info = super().update(trajectories, no_update_actor)
        if self.train_guidance_model and not no_update_actor:
            guidance_info = self.train_guidance(trajectories)
            info.update(guidance_info)
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
