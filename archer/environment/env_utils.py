import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, RobertaModel
import torch.nn as nn
import numpy as np
# from tqdm.auto import tqdm  # Import here to ensure we use the right tqdm variant


def add_trajectory_reward(trajectory):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_reward = np.sum([d["reward"] for d in trajectory])
    for d in trajectory:
        d.update({"trajectory_reward": trajectory_reward})
    return trajectory

def add_mc_return(trajectory, gamma = 0.95):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_rewards = np.array([d["reward"] for d in trajectory]).reshape(1, -1)
    gamma_row = np.cumprod(np.ones((1, trajectory_rewards.shape[1]))*gamma)
    gamma_matrix = np.triu(gamma_row.reshape(1, -1 )/ gamma_row.reshape(-1, 1))
    mc_returns = np.sum(trajectory_rewards*gamma_matrix, axis = 1)
    for d, mc in zip(trajectory, mc_returns):
        d.update({"mc_return": mc})
    return trajectory

def batch_interact_environment(agent, tokenizer, env, num_trajectories,
                               post_f=lambda x: x, use_tqdm=True, decode_f=lambda x: x,
                               env_idx=None, max_steps=1):
    """
    In a batched way, interact with the environments to get a list of trajectories.
    Each trajectory is a list of steps where each step is a dict with the keys:
    "observation", "next_observation", "reward", "done", and "action".
    
    post_f: A function to add additional attributes to the trajectory.
    max_steps: Maximum number of actions per episode (default=40, assuming each action generates ~8 tokens
               and maximum token limit is 324).
    """
    bsize = env.bsize
    all_trajectories = []
    num_batches = num_trajectories // bsize
    
    # Use tqdm for outer loop (batches)
    batch_iterator = tqdm(range(num_batches), desc="Batch", disable=not use_tqdm)
    
    for _ in batch_iterator:
        trajectories = [[] for _ in range(bsize)]
        batch_obs = env.reset(idx=env_idx)
        batch_done = np.zeros(bsize, dtype=bool)
        
        # Use tqdm for inner loop (steps) with dynamic total
        pbar = tqdm(total=max_steps, desc="Steps", leave=False, disable=not use_tqdm)
        
        for steps in range(max_steps):
            if np.all(batch_done):
                break
                
            # Process ALL environments at once, regardless of done status
            # We'll filter out the done environments later
            actions = agent.get_action(batch_obs)
            
            # Step environment with all actions at once
            batch_return = env.step(decode_f(actions))
            # Process results
            for i, result in enumerate(batch_return):
                if result is None or batch_done[i]:
                    continue
                next_obs, r, done = result
                trajectories[i].append({
                    "observation": batch_obs[i],
                    "next_observation": next_obs,
                    "reward": r,
                    "done": done,
                    "action": actions[i]
                })
                batch_obs[i] = next_obs
                batch_done[i] = done
            
            # Update progress
            pbar.update(1)
            pbar.set_postfix({"Done": f"{np.sum(batch_done)}/{bsize}"})
        
        pbar.close()
        
        # Display progress in the batch iterator
        if trajectories and trajectories[0]:
            batch_iterator.set_postfix({
                "Avg Steps": np.mean([len(t) for t in trajectories]), 
                "Avg Reward": np.mean([sum(d["reward"] for d in t) for t in trajectories if t])
            })
        
        # Process completed trajectories
        processed_trajectories = []
        for traj in trajectories:
            if traj:  # Only process non-empty trajectories
                processed_trajectories.append(post_f(add_mc_return(add_trajectory_reward(traj))))
        
        all_trajectories.extend(processed_trajectories)
    
    return all_trajectories





