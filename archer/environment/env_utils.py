from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, RobertaModel
import torch.nn as nn
import numpy as np

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
                               env_idx=None, max_steps=1, template=None):
    """
    In a batched way, interact with the environments to get a list of trajectories.
    Heavily optimized version for SCoRe's single-step trajectories.
    
    Args:
        agent: The agent to use for generating actions
        tokenizer: Tokenizer for the agent
        env: Environment to interact with
        num_trajectories: Number of trajectories to collect
        post_f: Post-processing function to apply to trajectories
        use_tqdm: Whether to show progress bar
        decode_f: Function to decode actions
        env_idx: Index of environment to use
        max_steps: Maximum number of steps per trajectory
        template: Optional template to use for prompting (for SMART_SCoRe)
    """
    bsize = env.bsize  # Environment's batch size
    all_trajectories = []
    
    # Calculate optimal batch size - use max possible that fits in memory
    optimal_batch_size = bsize
    
    # Process in as few batches as possible
    num_batches = (num_trajectories + optimal_batch_size - 1) // optimal_batch_size  # Ceiling division
    
    # Set up progress tracking - only show overall progress, not per batch
    pbar = tqdm(total=num_trajectories, desc="Samples", disable=not use_tqdm)
    
    trajectories_collected = 0
    correct_total = 0
    
    # If a template is provided, update the agent's template
    if template is not None and hasattr(agent, 'update_template'):
        agent.update_template(template)
    
    for batch_idx in range(num_batches):
        # Calculate batch size for this iteration
        current_batch_size = min(optimal_batch_size, num_trajectories - trajectories_collected)
        
        # Reset batch of environments without printing
        batch_obs = env.reset(idx=env_idx)
        
        # For the second turn of SCoRe, the environment history already contains the correction prompts
        # Check if the environment has a method to get current histories
        if hasattr(env, 'get_current_histories'):
            batch_obs = env.get_current_histories()
        
        # Single step for each environment (SCoRe uses single-step trajectories)
        actions = agent.get_action(batch_obs)
        batch_results = env.step(decode_f(actions))
        
        # Process results
        trajectories = []
        total_correct = 0  
        
        for i, result in enumerate(batch_results):
            if result is None:
                continue
                
            next_obs, reward, done = result
            total_correct += int(reward) 
            
            traj = [{
                "observation": batch_obs[i],
                "next_observation": next_obs,
                "reward": reward,
                "done": done,
                "action": actions[i]
            }]
            trajectories.append(post_f(add_mc_return(add_trajectory_reward(traj))))

        all_trajectories.extend(trajectories)
        trajectories_collected += len(trajectories)
        correct_total += total_correct
        
        # Update progress with overall stats
        pbar.update(len(trajectories))
        
        # Calculate global accuracy so far
        accuracy = correct_total / trajectories_collected if trajectories_collected else 0
        pbar.set_postfix({
            "Reward": f"{accuracy}",  
            "Overall": f"{correct_total}/{trajectories_collected}"
        })
    
    pbar.close()
    return all_trajectories





