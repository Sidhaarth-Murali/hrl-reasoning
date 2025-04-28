from archer.environment import batch_interact_environment
from archer.data import DummyDataset, ReplayBuffer
from archer.prompts import format_math_prompt, format_math_self_correction_prompt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from archer.algorithms.archer import ArcherTrainer
from archer.algorithms.online_filteredbc import BCTrainer
from archer.algorithms.score import SCoReTrainer
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
                         # SMART_SCoRe additions
                         use_smart_corrections: bool = False,  # Whether to use dynamic correction instructions
                         correction_model_path: str = None,    # Path to a model for generating correction instructions
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
        trainer = SCoReTrainer(
            agent=agent,
            tokenizer=tokenizer,
            accelerator=accelerator,
            lm_lr=lm_lr,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            batch_size=batch_size
        )

    # For non-SCoRe agents, we use a replay buffer
    if agent_type.lower() != "score":
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
        # For SCoRe, we don't need a replay buffer (on-policy)
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
            # Collect training trajectories.
            print(f"Iter:{i}")
            
            # For SCoRe, we need to gather two-turn trajectories
            if agent_type.lower() == "score":
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
                
                # Generate second turn with self-correction instruction
                print("Generating second turn with self-correction instruction")
                
                # Process the second turn in batches that match the environment's batch size
                batch_size = env.bsize
                num_batches = (len(trajectories_turn1) + batch_size - 1) // batch_size
                trajectories_turn2 = []
                
                for batch_idx in range(num_batches):
                    # Calculate start and end indices for this batch
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(trajectories_turn1))
                    batch_size_actual = end_idx - start_idx
                    
                    print(f"Processing batch {batch_idx+1}/{num_batches} with {batch_size_actual} samples")
                    
                    # Process batch corrections efficiently
                    from archer.prompts.math import generate_smart_correction_prompt
                    
                    print(f"Batch {batch_idx+1}/{num_batches}: Processing {batch_size_actual} samples")
                    
                    # Collect problem-solution pairs in batch
                    batch_problems = []
                    batch_solutions = []
                    for env_idx in tqdm(range(batch_size_actual), desc="Preparing batch"):
                        traj = trajectories_turn1[start_idx + env_idx]
                        batch_problems.append(traj[0]["observation"])
                        batch_solutions.append(traj[0]["action"])
                    
                    # Process all corrections at once in batch
                    if use_smart_corrections and hasattr(env, 'correction_model'):
                        # Use batch smart correction prompt generation
                        print("Generating batch correction instructions...")
                        correction_prompts = generate_smart_correction_prompt(
                            problem=batch_problems,
                            solution=batch_solutions,
                            correction_model=env.correction_model,
                            tokenizer=tokenizer,
                            device=env.correction_model.device if hasattr(env, 'correction_model') else None
                        )
                    else:
                        # Use static template in batch
                        correction_prompts = [format_math_self_correction_prompt(p + s) for p, s in zip(batch_problems, batch_solutions)]
                    
                    # Process cleaning and reset environment
                    correction_template = []
                    for env_idx, correction_prompt in enumerate(tqdm(correction_prompts, desc="Resetting environments")):
                        # Reset env with the correction prompt
                        env.env_list[env_idx].reset(None)
                        env.env_list[env_idx].history = correction_prompt
                        correction_template.append(remove_repeated_phrases(correction_prompt) + "\n\nYour Improved Solution:")
                    
                    # Run batch interaction
                    batch_trajectories = batch_interact_environment(
                        agent=agent,
                        tokenizer=tokenizer,
                        env=env,
                        num_trajectories=batch_size_actual,
                        env_idx=None,
                        use_tqdm=True,
                        decode_f=decode_f,
                        template=correction_template
                    )
                    
                    # Add these trajectories to our total
                    trajectories_turn2.extend(batch_trajectories)
                    print(f"Completed batch {batch_idx+1}, total trajectories so far: {len(trajectories_turn2)}/{len(trajectories_turn1)}")

                # Verify we have processed all trajectories
                assert len(trajectories_turn2) == len(trajectories_turn1), \
                    f"Expected {len(trajectories_turn1)} second-turn trajectories, but got {len(trajectories_turn2)}"
                
                # Calculate turn 2 rewards
                turn2_rewards = [d[0]["trajectory_reward"] for d in trajectories_turn2]
                
                # Combine trajectory data for SCoRe (on-policy, we use these directly)
                score_trajectories = []
                for t1, t2, r1, r2 in zip(trajectories_turn1, trajectories_turn2, turn1_rewards, turn2_rewards):
                    # Create a combined trajectory record with both turns
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
                _ = trainer.update(score_trajectories, no_update_actor=(i < warmup_iter))
                
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
                
                if agent_type.lower() == "score":
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
                    
                    # Generate second turn with self-correction instruction
                    # Process the second turn in batches for evaluation
                    batch_size = eval_env.bsize
                    num_eval_batches = (len(eval_trajectories_turn1) + batch_size - 1) // batch_size
                    eval_trajectories_turn2 = []
                    
                    for batch_idx in range(num_eval_batches):
                        # Calculate start and end indices for this batch
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, len(eval_trajectories_turn1))
                        batch_size_actual = end_idx - start_idx
                        
                        print(f"Processing eval batch {batch_idx+1}/{num_eval_batches} with {batch_size_actual} samples")
                        
                        # For each environment in this batch, set up the self-correction prompt
                        for env_idx in range(batch_size_actual):
                            traj = eval_trajectories_turn1[start_idx + env_idx]
                            problem = traj[0]["observation"]
                            solution1 = traj[0]["action"]
                            
                            # Create self-correction prompt based on settings
                            if use_smart_corrections and hasattr(eval_env, 'generate_correction_instruction'):
                                # Use dynamic correction prompt for evaluation too
                                correction_prompt = eval_env.generate_correction_instruction(problem, solution1)
                            else:
                                # Use static template (default)
                                correction_prompt = format_math_self_correction_prompt(problem + solution1)
                            
                            # Reset env with the correction prompt
                            eval_env.env_list[env_idx].reset(None)
                            eval_env.env_list[env_idx].history = correction_prompt
                        
                        # Run batch interaction for this specific batch
                        batch_trajectories = batch_interact_environment(
                            agent=agent,
                            tokenizer=tokenizer,
                            env=eval_env,
                            num_trajectories=batch_size_actual,
                            env_idx=None,
                            use_tqdm=True,  # Show progress for all batches
                            decode_f=decode_f
                        )
                        
                        # Add these trajectories to our total
                        eval_trajectories_turn2.extend(batch_trajectories)
                        print(f"Completed eval batch {batch_idx+1}, total trajectories so far: {len(eval_trajectories_turn2)}/{len(eval_trajectories_turn1)}")

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
        if agent_type.lower() != "score":
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
            if agent_type.lower() == "score":
                info["current_stage"] = trainer.current_stage
                info["total_steps"] = trainer.total_steps
                if "best_train_reward_so_far" not in info:
                    info["best_train_reward_so_far"] = best_train_reward
            wandb.log(info)

    # At the end of training, print information about the best checkpoint
    if agent_type.lower() == "score" and accelerator.is_main_process:
        print(f"\nTraining completed. Best training reward: {best_train_reward:.4f}")
        print(f"Best checkpoint saved at: {best_checkpoint_path}")
        
    # return trainer