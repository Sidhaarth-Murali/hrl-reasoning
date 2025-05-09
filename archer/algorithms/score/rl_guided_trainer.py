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
        **kwargs
    ):
        super().__init__(agent, tokenizer, accelerator, **kwargs)
        self.train_guidance_model = train_guidance_model
        self.guidance_kl_coef = guidance_kl_coef

        # Initialize or clone guidance model
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
        """
        orig = self.agent.model
        self.agent.model = self.guidance_model
        try:
            log_probs = self.agent.get_log_prob(prompts, actions)
        finally:
            self.agent.model = orig
        return log_probs

    def generate_custom_guidance(self, problems, initial_solutions, batch_size: int = 64):
        """
        Generate tailored guidance hints for each (problem, initial_solution) pair.

        Args:
            problems (Union[str, List[str]]):  Single problem or list of problems.
            initial_solutions (Union[str, List[str]]):  Single draft solution or list of solutions.
            batch_size (int, optional):  # items processed per forward‑pass (controls VRAM).  Default = 8.

        Returns:
            Tuple[List[str], List[str]]  →  (analysis_prompts, guidance_texts)  **or**
            Tuple[str, str]              →  when a single problem/solution is given.
        """
        import torch
        from tqdm import tqdm

        # ---------- normalise inputs ----------
        is_batch = isinstance(problems, list)
        if not is_batch:
            problems, initial_solutions = [problems], [initial_solutions]

        device            = self.agent.model.device
        analysis_prompts  = []  # full text fed to model
        guidance_texts    = []  # model‑generated instructions only

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
                batch_sols  = initial_solutions[start:start + batch_size]

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

                # ----- tokenise -----
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                ).to(device)

                # ----- generate guidance text -----
                
                ctx = torch.enable_grad() if getattr(self, "train_guidance_model", False) else torch.no_grad()
                with ctx:
                    outputs = self.guidance_model.generate(
                        input_ids       = inputs["input_ids"],
                        attention_mask  = inputs["attention_mask"],
                        max_new_tokens  = 256,
                        temperature     = 0.7,
                        top_p           = 0.95,
                        do_sample       = True,
                        num_beams       = 2,
                        use_cache       = True,
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
            print(f"[generate_custom_guidance] Falling back due to error ⇒ {e}")
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
        """
        return self.guidance_get_log_prob(analysis_prompts, guidance_texts)

    def train_guidance(self, trajectories):
        if not self.train_guidance_model:
            return {}

        # Extract data
        problems = [t[0]['observation']    for t in trajectories]
        sols     = [t[0]['action_turn1']    for t in trajectories]
        raw_r    = [t[0]['reward_turn2']    for t in trajectories]

        # Generate prompts and hints (only once)
        analysis_prompts, guidance_texts = self.generate_custom_guidance(problems, sols)
        if not guidance_texts:
            return {'guidance_loss': 0.0}

        device = self.agent.model.device
        # Symmetric rewards: {0->-1,1->+1}
        shaped_r = [1 if r>0 else -1 for r in raw_r]
        rewards_tensor = torch.tensor(shaped_r, dtype=torch.float, device=device)

        # Log-probs under guidance model
        self.guidance_model.train() # Ensure model is in training mode for log_prob calculation
        logp_tensor = self.calculate_guidance_log_probs(analysis_prompts, guidance_texts)
        # policy gradient loss
        policy_loss = -torch.mean(logp_tensor * rewards_tensor)    
        kl_div = torch.tensor(0.0, device=device, requires_grad=False) # Corrected placeholder for KL divergence

        kl_loss = self.guidance_kl_coef * kl_div
        total_loss = policy_loss + kl_loss

        # Optimizer step
        self.guidance_optimizer.zero_grad()
        self.accelerator.backward(total_loss)
        self.accelerator.clip_grad_norm_(self.guidance_model.parameters(), self.max_grad_norm)
        self.guidance_optimizer.step()

        return {
            'guidance_loss': policy_loss.item(),
            'guidance_reward_mean': rewards_tensor.mean().item(),
        }

    def update(self, trajectories, no_update_actor=False):
        info = super().update(trajectories, no_update_actor)
        if self.train_guidance_model and not no_update_actor:
            guidance_info = self.train_guidance(trajectories)
            info.update(guidance_info)
        return info

    def save(self, path):
        super().save(path)
        gpath = path.replace('.pt', '_guidance.pt')
        torch.save({
            'model_state_dict': self.guidance_model.state_dict(),
            'optimizer_state_dict': getattr(self, 'guidance_optimizer', None).state_dict() if hasattr(self, 'guidance_optimizer') else None
        }, gpath)
        print(f"Saved guidance model to {gpath}")

    def load(self, path):
        agent = super().load(path)
        gpath = path.replace('.pt', '_guidance.pt')
        if os.path.exists(gpath):
            ckpt = torch.load(gpath)
            self.guidance_model.load_state_dict(ckpt['model_state_dict'])
            if hasattr(self, 'guidance_optimizer') and 'optimizer_state_dict' in ckpt:
                self.guidance_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print(f"Loaded guidance model from {gpath}")
        return agent
