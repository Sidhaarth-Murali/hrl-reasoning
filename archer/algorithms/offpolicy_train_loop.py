from archer.environment import batch_interact_environment
from archer.data import DummyDataset, ReplayBuffer
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from archer.algorithms.archer import ArcherTrainer
from archer.algorithms.online_filteredbc import BCTrainer
import wandb
import threading
import os
import torch
import time

def offpolicy_train_loop(env,
                         eval_env,
                         agent,
                         tokenizer,
                         accelerator,
                         warmup_iter: int = 20,
                         rollout_size: int = 16,
                         eval_size: int = 1,
                         batch_size: int = 2,
                         capacity: int = 500000,
                         iterations: int = 10,
                         epochs: int = 3,
                         grad_accum_steps: int = 1,
                         env_idx: int = None,
                         do_sample: bool = False,
                         temperature: float = 2.0,
                         critic_lr: float = 1e-3,
                         lm_lr: float = 1e-5,
                         gamma: float = 0.9,
                         tau: float = 0.1,
                         use_wandb: bool = False,
                         env_load_path: str = '',
                         actor_epochs: int = 3,
                         max_grad_norm: float = 0.01,
                         save_path: str = None,
                         save_freq: int = 25,
                         eval_freq: int = 25,
                         agent_type: str = "archer",
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
            
    agent.prepare()
    print(">>> Start Iterations")

    for i in tqdm(range(iterations)):
        if accelerator.is_main_process:
            # Collect training trajectories.
            trajectories = batch_interact_environment(agent=agent,
                                                      tokenizer=tokenizer,
                                                      env=env,
                                                      num_trajectories=rollout_size,
                                                      env_idx=env_idx,
                                                      use_tqdm=False,
                                                      decode_f=decode_f)
            # Calculate training reward metrics (using the trajectory reward).
            train_rewards = [d[0]["trajectory_reward"] for d in trajectories]
            info = {
                "train_reward.mean": np.mean(train_rewards),
                "train_reward.max": np.max(train_rewards),
                "train_reward.min": np.min(train_rewards)
            }

            # Periodically evaluate.
            if (i + 1) % eval_freq == 0:
                old_sample = agent.do_sample
                agent.do_sample = False
                eval_trajectories = batch_interact_environment(agent=agent,
                                                               tokenizer=tokenizer,
                                                               env=eval_env,
                                                               num_trajectories=max(eval_size, eval_env.bsize),
                                                               env_idx=env_idx,
                                                               use_tqdm=True,
                                                               decode_f=decode_f)
                agent.do_sample = old_sample
                eval_rewards = [d[0]["trajectory_reward"] for d in eval_trajectories]
                info.update({
                    "eval_reward.mean": np.mean(eval_rewards),
                    "eval_reward.max": np.max(eval_rewards),
                    "eval_reward.min": np.min(eval_rewards)
                })

            # Save trajectories and insert transitions into the replay buffer.
            all_trajectories += trajectories
            data = sum(trajectories, [])
            for t in data:
                replay_buffer.insert(**t)

            print(">>> Saving Replay Buffer")
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
            torch.save(all_trajectories, os.path.join(save_path, 'trajectories.pt'))
            print(">>> Saved Replay Buffer")
            time.sleep(15)
        else:
            info = {}

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
        
        # Log metrics to WandB along with the current iteration.
        if use_wandb and accelerator.is_main_process:
            info["iteration"] = i + 1
            wandb.log(info)

        # Save model checkpoint periodically.
        if (i + 1) % save_freq == 0 and save_path is not None and accelerator.is_main_process:
            print("Saving model checkpoint...")
            trainer.save(os.path.join(save_path, 'trainer.pt'))
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))

    # return trainer
