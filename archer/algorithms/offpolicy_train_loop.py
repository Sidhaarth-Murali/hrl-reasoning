from archer.environment import batch_interact_environment
from archer.data import DummyDataset, ReplayBuffer
from archer.prompts import format_math_prompt, format_math_self_correction_prompt, generate_smart_correction_prompt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from archer.algorithms.archer import ArcherTrainer
from archer.algorithms.online_filteredbc import BCTrainer
from archer.algorithms.score import SCoReTrainer, RLGuidedSCoReTrainer, BiLevelSCoReTrainer
import wandb
from tqdm import tqdm
import os
import torch
import time
from collections import OrderedDict

def remove_repeated_phrases(text):
    lines = text.split('\\n')  # Since it's stored with literal '\n' inside the string
    seen = OrderedDict()
    for line in lines:
        if line not in seen:
            seen[line] = None
    return '\\n'.join(seen.keys())

def build_correction_templates(problems, initial_solutions, guidance_hints):
    """
    Return a list of nicely‑formatted correction prompts.

    Each prompt contains:
        1. Problem
        2. Previous Solution
        3. Guidance (deduplicated)
        4. An explicit instruction to follow the guidance
        5. A place for the student's improved solution
    """
    templates = []
    for prob, sol, hint in zip(problems, initial_solutions, guidance_hints):
        cleaned_hint = remove_repeated_phrases(hint).strip()

        template = (
            f"### Problem\n{prob}\n\n"
            f"### Previous Solution\n{sol}\n\n"
            f"### Guidance\n{cleaned_hint}\n\n"
            "🔎 **Task — Follow the guidance above to correct your work. Think step-by-step and at the end of the Solution, when you give your final answer, write it in the form:**\n"
            "**Final Answer: The final answer is $answer$. I hope it is correct.**\n"
        )
        templates.append(template)

    return templates


def offpolicy_train_loop(env,
                         eval_env,
                         agent,
                         tokenizer,
                         accelerator,
                         grad_accum_steps: int = 8,
                         warmup_iter: int = 1,
                         rollout_size: int = 512,     # Increased to 512 for SCoRe
                         eval_size: int = 4,
                         batch_size: int = 512,       # Increased to 512 for SCoRe
                         capacity: int = 500000,
                         iterations: int = 3000,      # Set to 3000 for SCoRe as specified
                         epochs: int = 3,
                         env_idx: int = None,
                         do_sample: bool = True,
                         temperature: float = 1.0,    # Set to 1.0 for SCoRe
                         critic_lr: float = 1e-3,
                         lm_lr: float = 5e-6,         # Set to 5e-6 for SCoRe on MATH
                         gamma: float = 0.9,
                         tau: float = 0.1,
                         use_wandb: bool = False,
                         env_load_path: str = '',
                         actor_epochs: int = 3,
                         max_grad_norm: float = 1.0,
                         save_path: str = None,
                         save_freq: int = 25,
                         eval_freq: int = 25,
                         agent_type: str = "archer",
                         # SCoRe specific parameters
                         alpha: float = 10.0,         # Reward shaping coefficient
                         beta1: float = 0.01,         # KL coefficient for Stage II
                         beta2: float = 0.1,          # KL coefficient for Stage I
                         stage1_steps: int = 1500,    # Number of training steps for Stage I
                         stage2_steps: int = 1500,    # Number of training steps for Stage II
                         use_smart_corrections: bool = False,  # Whether to use dynamic correction instructions
                         correction_model_path: str = None,    # Path to a model for generating correction instructions
                         train_guidance_model: bool = False,   # Whether to train the guidance model
                         guidance_lr: float = 1e-6,            # Learning rate for the guidance model
                         guidance_model_path: str = None,      # Path to initialize guidance model
                         guidance_kl_coef: float = 0.05,       # KL coefficient for guidance model
                         # BiLevel SCoRe additions
                         value_model_name: str = "roberta-base",  # Model name for the value function
                         value_lr: float = 1e-5,                 # Learning rate for the value function
                         value_coef: float = 0.5,                # Coefficient for the value function term
                         stop_value_gradients: bool = False,     # Whether to stop gradients from value function
                         decode_f: callable = lambda x: x,
                         **kwargs):

    # Set up trainer based on the agent type.
    if agent_type.lower() in ["chai", "archer", "archer_llm"]:
        trainer = ArcherTrainer(agent=agent,
                                accelerator=accelerator,
                                tokenizer=tokenizer,
                                critic_lr=critic_lr,
                                lm_lr=lm_lr,
                                gamma=gamma,
                                tau=tau,
                                epochs=epochs,
                                actor_epochs=actor_epochs,
                                grad_accum_steps=grad_accum_steps,
                                max_grad_norm=max_grad_norm)
    elif agent_type.lower() == "online_filteredbc":
        trainer = BCTrainer(agent=agent,
                            tokenizer=tokenizer,
                            accelerator=accelerator,
                            lm_lr=lm_lr,
                            epochs=actor_epochs,
                            grad_accum_steps=grad_accum_steps,
                            max_grad_norm=max_grad_norm)
    elif agent_type.lower() == "score":
        # Initialize SCoRe trainer with pure REINFORCE + KL
        if train_guidance_model and use_smart_corrections:
            print("Initializing RL-Guided SCoRe trainer with REINFORCE for both agent and guidance model")
            
            # Load guidance model if provided
            guidance_model = None
            if guidance_model_path is not None:
                try:
                    from transformers import AutoModelForCausalLM
                    # Add memory optimizations for guidance model
                    guidance_model = AutoModelForCausalLM.from_pretrained(
                        guidance_model_path, 
                        device_map="auto",  # Enable automatic device mapping
                        torch_dtype=torch.float16,  # Use half precision
                        low_cpu_mem_usage=True,
                        use_flash_attention_2=True  # Enable flash attention if available
                    )
                    print(f"Successfully loaded guidance model from {guidance_model_path}")
                except Exception as e:
                    print(f"Error loading guidance model: {e}")
                    print("Will initialize guidance model from reference model")
            
            # Enable memory optimizations for the trainer
            trainer = RLGuidedSCoReTrainer(
                agent=agent,
                tokenizer=tokenizer,
                accelerator=accelerator,
                guidance_model=guidance_model,
                guidance_lr=guidance_lr,
                guidance_kl_coef=guidance_kl_coef,
                train_guidance_model=train_guidance_model,
                lm_lr=lm_lr,
                grad_accum_steps=grad_accum_steps * 2,  # Double gradient accumulation steps
                max_grad_norm=max_grad_norm,
                alpha=alpha,
                beta1=beta1,
                beta2=beta2,
                stage1_steps=stage1_steps,
                stage2_steps=stage2_steps,
                batch_size=batch_size // 2,  # Halve batch size
                use_gradient_checkpointing=True,  # Enable gradient checkpointing
                use_memory_efficient_attention=True,  # Enable memory efficient attention
                max_micro_batch=4  # Use smaller micro-batches
            )
            
            # Enable gradient checkpointing for the agent model if available
            if hasattr(agent.model, "gradient_checkpointing_enable"):
                agent.model.gradient_checkpointing_enable()
            
            # Set up memory efficient attention if available
            if hasattr(agent.model, "config"):
                if hasattr(agent.model.config, "use_memory_efficient_attention"):
                    agent.model.config.use_memory_efficient_attention = True
                if hasattr(agent.model.config, "use_flash_attention"):
                    agent.model.config.use_flash_attention = True
        else:
            # Standard SCoRe trainer with memory optimizations
            trainer = SCoReTrainer(
                agent=agent,
                tokenizer=tokenizer,
                accelerator=accelerator,
                lm_lr=lm_lr,
                grad_accum_steps=grad_accum_steps * 2,  # Double gradient accumulation steps
                max_grad_norm=max_grad_norm,
                alpha=alpha,
                beta1=beta1,
                beta2=beta2,
                stage1_steps=stage1_steps,
                stage2_steps=stage2_steps,
                batch_size=batch_size // 2,  # Halve batch size
                use_gradient_checkpointing=True,  # Enable gradient checkpointing
                use_memory_efficient_attention=True  # Enable memory efficient attention
            )
            
            # Enable gradient checkpointing for the agent model if available
            if hasattr(agent.model, "gradient_checkpointing_enable"):
                agent.model.gradient_checkpointing_enable()
            
            # Set up memory efficient attention if available
            if hasattr(agent.model, "config"):
                if hasattr(agent.model.config, "use_memory_efficient_attention"):
                    agent.model.config.use_memory_efficient_attention = True
                if hasattr(agent.model.config, "use_flash_attention"):
                    agent.model.config.use_flash_attention = True
    elif agent_type.lower() == "bi_level_score":
        print("Initializing BiLevel SCoRe trainer with memory optimizations")
        
        # Load guidance model if provided
        guidance_model = None
        if guidance_model_path is not None:
            try:
                from transformers import AutoModelForCausalLM, BitsAndBytesConfig
                # Add memory optimizations for guidance model
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_skip_modules=None,
                )
                guidance_model = AutoModelForCausalLM.from_pretrained(
                    guidance_model_path, 
                    device_map="auto",  # Enable automatic device mapping
                    torch_dtype=torch.float16,  # Use half precision
                    low_cpu_mem_usage=True,
                    use_flash_attention_2=True,  # Enable flash attention if available
                    quantization_config=quantization_config  # Enable 8-bit quantization
                )
                print(f"Successfully loaded guidance model from {guidance_model_path} with memory optimizations")
            except Exception as e:
                print(f"Error loading guidance model with full optimizations: {e}")
                print("Trying fallback loading with basic optimizations")
                try:
                    guidance_model = AutoModelForCausalLM.from_pretrained(
                        guidance_model_path,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True
                    )
                    print("Loaded guidance model with basic memory optimizations")
                except Exception as e:
                    print(f"Error loading guidance model: {e}")
                    print("Will initialize guidance model from reference model")
        
        # Create BiLevel SCoRe trainer with memory optimizations
        trainer = BiLevelSCoReTrainer(
            agent=agent,
            tokenizer=tokenizer,
            accelerator=accelerator,
            guidance_model=guidance_model,
            guidance_lr=guidance_lr,
            guidance_kl_coef=guidance_kl_coef,
            train_guidance_model=train_guidance_model,
            value_model_name=value_model_name,
            value_lr=value_lr,
            value_coef=value_coef,
            stop_value_gradients=stop_value_gradients,
            cache_dir=kwargs.get("cache_dir", None),
            lm_lr=lm_lr,
            grad_accum_steps=grad_accum_steps * 2,  # Double gradient accumulation steps
            max_grad_norm=max_grad_norm,
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            batch_size=batch_size // 2,  # Halve batch size
            use_gradient_checkpointing=True,  # Enable gradient checkpointing
            use_memory_efficient_attention=True,  # Enable memory efficient attention
            max_micro_batch=4  # Use smaller micro-batches
        )
        
        # Enable memory optimizations for agent model
        if hasattr(agent.model, "gradient_checkpointing_enable"):
            agent.model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing for agent model")
        
        # Set up memory efficient attention for agent model
        if hasattr(agent.model, "config"):
            if hasattr(agent.model.config, "use_memory_efficient_attention"):
                agent.model.config.use_memory_efficient_attention = True
                print("Enabled memory efficient attention for agent model")
            if hasattr(agent.model.config, "use_flash_attention"):
                agent.model.config.use_flash_attention = True
                print("Enabled flash attention for agent model")

    # For non-SCoRe agents, we use a replay buffer
    if agent_type.lower() not in ["score", "bi_level_score"]:
        replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
        all_trajectories = []
        
        if accelerator.is_main_process:
            if os.path.exists(os.path.join(save_path, 'trainer.pt')):
                print("Loading from checkpoint")
                trainer.load(os.path.join(save_path, 'trainer.pt'))
                all_trajectories = torch.load(os.path.join(save_path, 'trajectories.pt'))
                replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'))
            else:
                print("Creating new checkpoint directory")
                os.makedirs(save_path, exist_ok=True)
    else:
        # For SCoRe and BiLevel SCoRe, we don't need a replay buffer (on-policy)
        # Initialize variables for tracking best training reward
        best_train_reward = -float('inf')
        best_checkpoint_path = os.path.join(save_path, 'best_train_reward_checkpoint.pt')
        
        if accelerator.is_main_process:
            if os.path.exists(os.path.join(save_path, 'trainer.pt')):
                print("Loading from checkpoint")
                trainer.load(os.path.join(save_path, 'trainer.pt'))
                # Try to load best reward info if it exists
                if os.path.exists(os.path.join(save_path, 'best_train_reward.pt')): 
                    best_train_reward = torch.load(os.path.join(save_path, 'best_train_reward.pt'))
                    print(f"Loaded previous best training reward: {best_train_reward:.4f}")
            else:
                print("Creating new checkpoint directory")
                os.makedirs(save_path, exist_ok=True)
            
    accelerator.unwrap_model(agent).prepare()
    print(">>> Start Iterations")
    for i in tqdm(range(iterations)):
        if accelerator.is_main_process:
            
            # For SCoRe and BiLevel SCoRe, we need to gather two-turn trajectories
            if agent_type.lower() in ["score", "bi_level_score"]:
                # First turn generation - use zero-shot math template
                from archer.prompts.math import MATH_ZERO_SHOT_TEMPLATE
                trajectories_turn1 = batch_interact_environment(
                    agent=agent,
                    tokenizer=tokenizer,
                    env=env,
                    num_trajectories=rollout_size,
                    env_idx=env_idx,
                    use_tqdm=True,
                    decode_f=decode_f,
                    template=MATH_ZERO_SHOT_TEMPLATE
                )                
                # Calculate turn 1 rewards
                turn1_rewards = [d[0]["trajectory_reward"] for d in trajectories_turn1]
                
                # NEW CODE: Use trainer's guidance generation functionality
                problems = [traj[0]['observation'] for traj in trajectories_turn1]
                solutions = [traj[0]['action'] for traj in trajectories_turn1]

                # Generate guidance via trainer's custom method
                print("Generating guidance via trainer's custom method")
                
                # Check if trainer has custom guidance method
                if hasattr(trainer, 'generate_custom_guidance'):
                    try:
                        analysis_prompts, guidance_hints = trainer.generate_custom_guidance(
                            problems, solutions
                        )
                    except Exception as e:
                        print(f"Error using trainer's custom guidance: {e}")
                        print("Falling back to generate_smart_correction_prompt")
                        # Import the fallback function
                        from archer.prompts import generate_smart_correction_prompt
                        # Use the fallback function
                        analysis_prompts = generate_smart_correction_prompt(
                            problems, solutions,
                            correction_model=None  # Use static template path
                        )
                        guidance_hints = ["" for _ in analysis_prompts]
                else:
                    print("Trainer does not have custom guidance method, using generate_smart_correction_prompt")
                    # Import the fallback function
                    from archer.prompts import generate_smart_correction_prompt
                    # Use the fallback function
                    analysis_prompts = generate_smart_correction_prompt(
                        problems, solutions,
                        correction_model=None  # Use static template path
                    )
                    guidance_hints = ["" for _ in analysis_prompts]

                # Format correction templates
                correction_templates = build_correction_templates(problems, solutions, guidance_hints)

                # Generate second turn with custom guidance
                print("Generating second turn with custom guidance")
                trajectories_turn2 = batch_interact_environment(
                    agent=agent,
                    tokenizer=tokenizer,
                    env=env,
                    num_trajectories=len(trajectories_turn1),
                    env_idx=None,
                    use_tqdm=True,
                    decode_f=decode_f,
                    template=correction_templates
                )

                # Verify we have processed all trajectories
                assert len(trajectories_turn2) == len(trajectories_turn1), \
                    f"Expected {len(trajectories_turn1)} second-turn trajectories, but got {len(trajectories_turn2)}"
                
                # Calculate turn 2 rewards
                turn2_rewards = [d[0]["trajectory_reward"] for d in trajectories_turn2]
                
                # Combine trajectory data for SCoRe (on-policy, we use these directly)
                score_trajectories = []
                
                # Process trajectories in smaller batches to save memory
                batch_size = 16  # Smaller batch size for processing
                for batch_start in range(0, len(trajectories_turn1), batch_size):
                    batch_end = min(batch_start + batch_size, len(trajectories_turn1))
                    batch_t1 = trajectories_turn1[batch_start:batch_end]
                    batch_t2 = trajectories_turn2[batch_start:batch_end]
                    batch_r1 = turn1_rewards[batch_start:batch_end]
                    batch_r2 = turn2_rewards[batch_start:batch_end]
                    
                    # Clear CUDA cache before processing each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    for t1, t2, r1, r2 in zip(batch_t1, batch_t2, batch_r1, batch_r2):
                        # Move data to CPU to save GPU memory
                        traj_data = [{
                            "observation": t1[0]["observation"],
                            "action": t1[0]["action"],
                            "action_turn1": t1[0]["action"],
                            "action_turn2": t2[0]["action"],
                            "reward": t1[0]["reward"],
                            "reward_turn1": t1[0]["reward"],
                            "reward_turn2": t2[0]["reward"],
                            "next_observation": t1[0]["next_observation"],
                            "done": t1[0]["done"],
                            "mc_return": t1[0]["mc_return"],
                            "trajectory_reward": (t1[0]["trajectory_reward"] + t2[0]["trajectory_reward"]) / 2.0
                        }]
                        score_trajectories.append(traj_data)
                    
                    # Explicitly delete batch data to free memory
                    del batch_t1, batch_t2, batch_r1, batch_r2
                
                # Calculate combined training reward metrics
                train_rewards = [d[0]["trajectory_reward"] for d in score_trajectories]
                current_train_reward = np.mean(train_rewards)
                turn1_mean = np.mean(turn1_rewards)
                turn2_mean = np.mean(turn2_rewards)
                info = {
                    "train_reward.mean": current_train_reward,
                    "train_reward.max": np.max(train_rewards),
                    "train_reward.min": np.min(train_rewards),
                    "train_reward_turn1.mean": turn1_mean,
                    "train_reward_turn2.mean": turn2_mean,
                    "train_reward_improvement": turn2_mean - turn1_mean
                }
                
                # For SCoRe: on-policy update with the freshly collected trajectories
                print("Training with on-policy trajectories (SCoRe)")
                
                # Process trajectories in smaller batches during update
                update_batch_size = 8  # Even smaller batch size for model update
                for update_start in range(0, len(score_trajectories), update_batch_size):
                    update_end = min(update_start + update_batch_size, len(score_trajectories))
                    batch_trajectories = score_trajectories[update_start:update_end]
                    
                    # Clear CUDA cache before each update batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    try:
                        _ = trainer.update(batch_trajectories, no_update_actor=(i < warmup_iter))
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            print(f"OOM in batch {update_start}-{update_end}, reducing batch size and retrying")
                            # Try with even smaller batch
                            for single_traj in batch_trajectories:
                                _ = trainer.update([single_traj], no_update_actor=(i < warmup_iter))
                        else:
                            raise e
                
                # Check if current reward is the best so far
                if current_train_reward > best_train_reward:
                    best_train_reward = current_train_reward
                    print(f"New best training reward: {best_train_reward:.4f}, saving checkpoint")
                    # Save best checkpoint
                    trainer.save(best_checkpoint_path)
                    # Also save the best reward value for resuming training
                    torch.save(best_train_reward, os.path.join(save_path, 'best_train_reward.pt'))
                
                # Also save periodic checkpoints
                if (i + 1) % save_freq == 0 and save_path is not None:
                    print(f"Saving periodic checkpoint (iter {i+1})")
                    trainer.save(os.path.join(save_path, f'checkpoint_{i+1}.pt'))
                    # Always save latest as trainer.pt for resuming
                    trainer.save(os.path.join(save_path, 'trainer.pt'))
                
            else:
                # Standard single-turn trajectory collection for other agent types
                trajectories = batch_interact_environment(
                    agent=agent,
                    tokenizer=tokenizer,
                    env=env,
                    num_trajectories=rollout_size,
                    env_idx=env_idx,
                    use_tqdm=True,
                    decode_f=decode_f
                )
                
                # Calculate training reward metrics
                train_rewards = [d[0]["trajectory_reward"] for d in trajectories]
                info = {
                    "train_reward.mean": np.mean(train_rewards),
                    "train_reward.max": np.max(train_rewards),
                    "train_reward.min": np.min(train_rewards)
                }
                
                # Save trajectories to replay buffer for off-policy methods
                all_trajectories += trajectories
                data = sum(trajectories, [])
                for t in data:
                    replay_buffer.insert(**t)

                print(">>> Saving Replay Buffer")
                torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
                torch.save(all_trajectories, os.path.join(save_path, 'trajectories.pt'))
                print(">>> Saved Replay Buffer")
                
            print(info)

            # Periodically evaluate.
            if (i + 1) % eval_freq == 0:
                old_sample = agent.do_sample
                agent.do_sample = False
                
                if agent_type.lower() in ["score", "bi_level_score"]:
                    # First turn evaluation
                    eval_trajectories_turn1 = batch_interact_environment(
                        agent=agent,
                        tokenizer=tokenizer,
                        env=eval_env,
                        num_trajectories=max(eval_size, eval_env.bsize),
                        env_idx=env_idx,
                        use_tqdm=True,
                        decode_f=decode_f
                    )
                    
                    # Calculate turn 1 rewards
                    eval_turn1_rewards = [d[0]["trajectory_reward"] for d in eval_trajectories_turn1]
                    
                    # NEW CODE: Use trainer's guidance generation for evaluation too
                    problems_eval = [traj[0]['observation'] for traj in eval_trajectories_turn1]
                    solutions_eval = [traj[0]['action'] for traj in eval_trajectories_turn1]
                    
                    # Generate evaluation guidance
                    eval_prompts, eval_hints = trainer.generate_custom_guidance(
                        problems_eval, solutions_eval
                    )
                    
                    # Format evaluation correction templates
                    eval_templates = [
                        remove_repeated_phrases(hint) + "\n\nYour Improved Solution:"
                        for hint in eval_hints
                    ]
                    
                    # Run batch interaction for evaluation second turn
                    eval_trajectories_turn2 = batch_interact_environment(
                        agent=agent,
                        tokenizer=tokenizer,
                        env=eval_env,
                        num_trajectories=len(eval_trajectories_turn1),
                        env_idx=None,
                        use_tqdm=True,
                        decode_f=decode_f,
                        template=eval_templates
                    )
                    
                    # Verify we have processed all trajectories
                    assert len(eval_trajectories_turn2) == len(eval_trajectories_turn1), \
                        f"Expected {len(eval_trajectories_turn1)} second-turn eval trajectories, but got {len(eval_trajectories_turn2)}"
                    
                    # Calculate turn 2 rewards
                    eval_turn2_rewards = [d[0]["trajectory_reward"] for d in eval_trajectories_turn2]
                    
                    # Calculate evaluation metrics
                    eval_turn1_mean = np.mean(eval_turn1_rewards)
                    eval_turn2_mean = np.mean(eval_turn2_rewards)
                    
                    info.update({
                        "eval_reward_turn1.mean": eval_turn1_mean,
                        "eval_reward_turn2.mean": eval_turn2_mean,
                        "eval_reward_improvement": eval_turn2_mean - eval_turn1_mean,
                        "best_train_reward_so_far": best_train_reward  # Track best reward in logs
                    })
                else:
                    # Standard single-turn evaluation for other agent types
                    eval_trajectories = batch_interact_environment(
                        agent=agent,
                        tokenizer=tokenizer,
                        env=eval_env,
                        num_trajectories=max(eval_size, eval_env.bsize),
                        env_idx=env_idx,
                        use_tqdm=True,
                        decode_f=decode_f
                    )
                    
                    eval_rewards = [d[0]["trajectory_reward"] for d in eval_trajectories]
                    info.update({
                        "eval_reward.mean": np.mean(eval_rewards),
                        "eval_reward.max": np.max(eval_rewards),
                        "eval_reward.min": np.min(eval_rewards)
                    })
                
                agent.do_sample = old_sample
        else:
            info = {}

        # For non-SCoRe agents, we need to sync the replay buffer across processes
        if agent_type.lower() not in ["score", "bi_level_score"]:
            accelerator.wait_for_everyone()

            # Reload trajectories and replay buffer to ensure consistency.
            all_trajectories = torch.load(os.path.join(save_path, 'trajectories.pt'))
            replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'))
            print("Training")

            # Use a filtered buffer if needed.
            if 'filtered' in agent_type.lower():
                filtered_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
                episode_rewards = [d[0]["trajectory_reward"] for d in all_trajectories]
                cutoff = np.quantile(episode_rewards, 1 - 0.1)
                print("Episode Reward Cutoff: ", cutoff)
                filtered_trajectories = list(filter(lambda x: x[0]["trajectory_reward"] >= cutoff, all_trajectories))
                data = sum(filtered_trajectories, [])
                for d in data:
                    filtered_buffer.insert(**d)
                _ = trainer.update(filtered_buffer, no_update_actor=(i < warmup_iter))
            else:
                _ = trainer.update(replay_buffer, no_update_actor=(i < warmup_iter))
            
            # Save model checkpoint periodically for non-SCoRe methods
            if (i + 1) % save_freq == 0 and save_path is not None and accelerator.is_main_process:
                print("Saving model checkpoint...")
                trainer.save(os.path.join(save_path, 'trainer.pt'))
                torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
        
        # Log metrics to WandB along with the current iteration.
        if use_wandb and accelerator.is_main_process:
            info["iteration"] = i + 1
            if agent_type.lower() in ["score", "bi_level_score"]:
                info["current_stage"] = trainer.current_stage
                info["total_steps"] = trainer.total_steps
                if "best_train_reward_so_far" not in info:
                    info["best_train_reward_so_far"] = best_train_reward
            wandb.log(info)

    # At the end of training, print information about the best checkpoint
    if agent_type.lower() in ["score", "bi_level_score"] and accelerator.is_main_process:
        print(f"\nTraining completed. Best training reward: {best_train_reward:.4f}")
        print(f"Best checkpoint saved at: {best_checkpoint_path}")
        
    # return trainer
    return trainer