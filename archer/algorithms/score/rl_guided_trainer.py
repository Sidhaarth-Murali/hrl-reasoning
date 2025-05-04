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
    def __init__(self, 
                 agent,
                 tokenizer,
                 accelerator,
                 guidance_model=None,
                 guidance_lr: float = 1e-6,
                 guidance_kl_coef: float = 0.05,
                 train_guidance_model: bool = True,
                 **kwargs):
        """
        Initialize RLGuidedSCoReTrainer with additional parameters for guidance model.
        """
        super().__init__(agent, tokenizer, accelerator, **kwargs)
        
        self.train_guidance_model = train_guidance_model
        self.guidance_kl_coef = guidance_kl_coef
        
        # If guidance_model is provided, use it; otherwise, use the ref_model
        if guidance_model is not None:
            self.guidance_model = guidance_model
        else:
            # Copy reference model to use as guidance model
            print("Creating guidance model from reference model...")
            
            # Check if we're using a LoRA model (PeftModel)
            from peft import PeftModel
            if isinstance(self.ref_model, PeftModel):
                # For LoRA models, we need to clone it differently
                print("Detected LoRA model, creating copy with shared weights...")
                # Create a copy that shares the base weights but has separate LoRA weights
                from peft import get_peft_model_state_dict
                
                # Create a copy with the same base model and config
                base_model = self.ref_model.get_base_model()
                peft_config = self.ref_model.peft_config
                
                # Create a new instance of PeftModel with the same base model
                import copy
                self.guidance_model = copy.deepcopy(self.ref_model)
                
                # Move to the desired device
                self.guidance_model = self.guidance_model.to(agent.model.device)
                
                # Force memory cleanup
                torch.cuda.empty_cache()
            else:
                # Standard model (not LoRA)
                # Use state_dict method instead of deepcopy to avoid CUDA OOM
                self.guidance_model = type(self.ref_model)(**{
                    k: v for k, v in self.ref_model.config.to_dict().items() 
                    if not k.startswith('_')
                }).to(agent.model.device)
                
                # Load state dict from CPU to save memory
                ref_state_dict = {k: v.to('cpu') for k, v in self.ref_model.state_dict().items()}
                self.guidance_model.load_state_dict(ref_state_dict)
                del ref_state_dict
                torch.cuda.empty_cache()  # Clean up any temp tensors
            
        # Initialize guidance model optimizer if training is enabled
        if self.train_guidance_model:
            # For memory efficiency, we'll use Adam with low learning rate
            self.guidance_optimizer = torch.optim.Adam(
                self.guidance_model.parameters(), 
                lr=guidance_lr
            )
            self.guidance_optimizer = self.accelerator.prepare(self.guidance_optimizer)
            
            # Create a frozen reference model for the guidance model's KL divergence
            print("Creating frozen reference guidance model...")
            
            # For KL divergence, we only need a lightweight snapshot of parameters
            # Rather than creating another full model, just save the current parameter values
            self.ref_params = {}
            for name, param in self.guidance_model.named_parameters():
                if param.requires_grad:
                    self.ref_params[name] = param.detach().cpu().clone()
            
            print(f"Guidance model training enabled with lr={guidance_lr}")
        else:
            print("Guidance model training disabled")
            
    def generate_custom_guidance(self, problems, initial_solutions):
        """
        Generate custom guidance using the guidance model.
        Returns the guidance text and the log probabilities.
        """
        device = self.agent.model.device
        guidance_texts = []
        log_probs = []
        
        batch_size = 2  # Reduced from 4 to 2 for better memory efficiency
        
        for i in range(0, len(problems), batch_size):
            batch_problems = problems[i:i+batch_size]
            batch_solutions = initial_solutions[i:i+batch_size]
            
            # Create analysis prompts
            analysis_prompts = []
            for p, s in zip(batch_problems, batch_solutions):
                prompt = f"""You are an expert math tutor reviewing a student's solution to a math problem.\n\n
                PROBLEM:{p}
                \n\n
                INITIAL SOLUTION:{s}
                \n\n
                PROMPT: First, analyze the solution for errors or misconceptions. Then, write a brief, helpful instruction that will guide the student toward correcting their solution.
                Your instruction should be specific to the errors you identified, but don't solve the problem for them.
                Your response should be ONLY the instruction for the student to improve their solution, nothing else. DO NOT include ANY SOLUTION.
                
                GUIDING INSTRUCTION:
                """
                analysis_prompts.append(prompt)
            
            # Tokenize inputs
            self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = self.tokenizer(
                analysis_prompts, 
                return_tensors="pt", 
                truncation=True,
                padding=True,
            ).to(device)
            
            # Track log probabilities if training is enabled
            if self.train_guidance_model:
                # Generate with the guidance model and track log probabilities
                try:
                    with torch.set_grad_enabled(True):
                        # Use a smaller max_new_tokens to save memory
                        outputs = self.guidance_model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=128,  # Reduced from 256/512
                            temperature=0.7,
                            do_sample=True,
                            use_cache=True,
                            num_beams=1,  # Reduced from 2 for memory efficiency
                            output_scores=True,
                            return_dict_in_generate=True
                        )
                        
                        # Extract sequences and scores
                        sequences = outputs.sequences
                        scores = outputs.scores
                        
                        # Calculate log probabilities for REINFORCE
                        batch_log_probs = []
                        for j in range(len(batch_problems)):
                            # Get input length to know where generated part starts
                            input_length = len(inputs["input_ids"][j])
                            
                            # Compute log probability of generated sequence
                            seq_log_prob = 0.0
                            for k, logits in enumerate(scores):
                                if j < logits.shape[0]:  # Make sure we don't go out of bounds
                                    token_idx = sequences[j, input_length + k].item()
                                    token_logits = logits[j]
                                    token_log_prob = F.log_softmax(token_logits, dim=-1)[token_idx]
                                    seq_log_prob += token_log_prob
                                
                            batch_log_probs.append(seq_log_prob.detach())
                        
                        log_probs.extend(batch_log_probs)
                    
                except torch.cuda.OutOfMemoryError:
                    print("Warning: OOM during guidance generation with gradients. Falling back to no gradients.")
                    # Fall back to generation without tracking probabilities
                    with torch.no_grad():
                        outputs = self.guidance_model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=128,
                            temperature=0.7,
                            do_sample=True,
                            use_cache=True
                        )
                        sequences = outputs
                        # Add dummy log probs
                        log_probs.extend([torch.tensor(0.0, device=device) for _ in range(len(batch_problems))])
            else:
                # Just generate without tracking probabilities
                with torch.no_grad():
                    outputs = self.guidance_model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=128,  # Reduced from 256
                        temperature=0.7,
                        do_sample=True,
                        use_cache=True,
                        num_beams=1  # Reduced from 2
                    )
                    sequences = outputs
            
            # Decode outputs
            input_lengths = [len(ids) for ids in inputs["input_ids"]]
            for j, output in enumerate(sequences):
                if hasattr(outputs, 'sequences'):
                    output = outputs.sequences[j]
                    
                # Extract only the generated part
                instruction = self.tokenizer.decode(
                    output[input_lengths[j]:], 
                    skip_special_tokens=True
                ).strip()
                
                # Format the final prompt with the guidance
                final_prompt = (batch_problems[j] + batch_solutions[j] + 
                               "\n\nSuggestive Correction: \n\n" + instruction)
                guidance_texts.append(final_prompt)
                
            # Clean up GPU memory
            del inputs, outputs
            if 'sequences' in locals():
                del sequences
            if 'scores' in locals():
                del scores
            torch.cuda.empty_cache()
            
        return guidance_texts, log_probs
    
    def train_guidance_model(self, trajectories):
        """
        Train the guidance model using REINFORCE based on the final rewards.
        """
        if not self.train_guidance_model:
            return {}
            
        print(">>> Training guidance model...")
        
        # Extract the relevant data from trajectories
        problems = [traj[0]["observation"] for traj in trajectories]
        initial_solutions = [traj[0]["action_turn1"] for traj in trajectories]
        rewards_turn2 = [traj[0]["reward_turn2"] for traj in trajectories]
        
        # Get guidance and log probabilities for REINFORCE
        guidance_prompts, log_probs = self.generate_custom_guidance(problems, initial_solutions)
        
        # Skip if no log probabilities were collected
        if not log_probs:
            return {"guidance_loss": 0.0}
        
        # Handle log probabilities safely, some might be dummies from OOM fallback
        device = self.agent.model.device
        try:
            # Try to stack the log probs if they're compatible
            log_probs_tensor = torch.stack(log_probs)
        except (RuntimeError, TypeError):
            # If stacking fails (e.g., due to different shapes or dtypes)
            print("Warning: Could not stack log probs. Using dummy values.")
            log_probs_tensor = torch.zeros(len(log_probs), device=device)
            
        # Convert rewards to tensor
        rewards_tensor = torch.tensor(rewards_turn2, device=device)
        
        # Compute KL divergence with reference guidance model
        try:
            kl_div = self.calculate_guidance_kl_divergence(problems, initial_solutions)
        except (RuntimeError, ValueError, TypeError) as e:
            print(f"Warning: KL calculation failed: {e}. Using zero KL.")
            kl_div = torch.tensor(0.0, device=device, requires_grad=True)
        
        # REINFORCE loss
        policy_loss = -torch.mean(log_probs_tensor * rewards_tensor)
        kl_loss = self.guidance_kl_coef * kl_div
        total_loss = policy_loss + kl_loss
        
        # Backward pass and optimization
        self.guidance_optimizer.zero_grad()
        
        try:
            self.accelerator.backward(total_loss)
            self.accelerator.clip_grad_norm_(self.guidance_model.parameters(), self.max_grad_norm)
            self.guidance_optimizer.step()
        except RuntimeError as e:
            print(f"Warning: Backward pass failed: {e}. Skipping update.")
            return {
                "guidance_loss": 0.0,
                "guidance_policy_loss": 0.0,
                "guidance_kl_loss": 0.0,
                "guidance_reward_mean": rewards_tensor.mean().item() if rewards_tensor.numel() > 0 else 0.0
            }
        
        return {
            "guidance_loss": total_loss.item(),
            "guidance_policy_loss": policy_loss.item(),
            "guidance_kl_loss": kl_loss.item(),
            "guidance_reward_mean": rewards_tensor.mean().item()
        }
    
    def calculate_guidance_kl_divergence(self, problems, initial_solutions):
        """Calculate KL divergence based on parameter differences instead of logits.
        This is much more memory-efficient than creating a full reference model.
        """
        # If reference parameters are not available, return zero tensor
        if not hasattr(self, 'ref_params') or not self.ref_params:
            return torch.tensor(0.0, device=self.agent.model.device, requires_grad=True)
            
        # Calculate KL as the sum of squared parameter differences (weighted L2 norm)
        kl_div = 0.0
        device = self.agent.model.device
        
        # Process parameters in small batches to save memory
        for name, param in self.guidance_model.named_parameters():
            if param.requires_grad and name in self.ref_params:
                # Move reference param to device temporarily
                ref_param = self.ref_params[name].to(device)
                
                # Calculate parameter difference and squared L2 norm
                param_diff = param - ref_param
                param_kl = torch.sum(param_diff ** 2)
                
                # Weight the KL by parameter size
                weighted_kl = param_kl / param.numel()
                kl_div = kl_div + weighted_kl
                
                # Move reference back to CPU
                self.ref_params[name] = ref_param.cpu()
                del ref_param
                
        # Normalize by number of parameters
        num_params = len([p for p in self.guidance_model.parameters() if p.requires_grad])
        if num_params > 0:
            kl_div = kl_div / num_params
            
        return kl_div
    
    def update(self, trajectories, no_update_actor=False):
        """
        Override the update method to also train the guidance model.
        """
        # First, let the parent class update the SCoRe model
        info = super().update(trajectories, no_update_actor)
        
        # Then, update the guidance model if enabled
        if self.train_guidance_model and not no_update_actor:
            guidance_info = self.train_guidance_model(trajectories)
            info.update(guidance_info)
        
        return info
        
    def save(self, path):
        """Save both models' state."""
        super().save(path)
        
        # Save guidance model separately
        guidance_path = path.replace('.pt', '_guidance.pt')
        torch.save({
            'model_state_dict': self.guidance_model.state_dict(),
            'optimizer_state_dict': self.guidance_optimizer.state_dict() if hasattr(self, 'guidance_optimizer') else None
        }, guidance_path)
        print(f"Saved guidance model to {guidance_path}")
        
    def load(self, path):
        """Load both models' state."""
        breakpoint()
        agent = super().load(path)
        
        # Load guidance model if available
        guidance_path = path.replace('.pt', '_guidance.pt')
        if os.path.exists(guidance_path):
            try:
                checkpoint = torch.load(guidance_path)
                self.guidance_model.load_state_dict(checkpoint['model_state_dict'])
                if hasattr(self, 'guidance_optimizer') and 'optimizer_state_dict' in checkpoint:
                    self.guidance_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Loaded guidance model from {guidance_path}")
            except Exception as e:
                print(f"Error loading guidance model: {e}")
                
        return agent 