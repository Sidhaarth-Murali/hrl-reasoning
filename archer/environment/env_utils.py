from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, RobertaModel
import torch.nn as nn
import numpy as np
import torch  # Add explicit torch import

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
                               env_idx=None, max_steps=1, template=None, bsize=None):
    """
    Interact with a batch of environments.
    
    Args:
        agent: The agent to interact with the environment.
        tokenizer: The tokenizer for the agent.
        env: The environment to interact with.
        num_trajectories: The number of trajectories to collect.
        post_f: A function to apply to each trajectory after collection.
        use_tqdm: Whether to use tqdm for progress reporting.
        decode_f: A function to decode the agent's actions.
        env_idx: The index of the environment to use.
        max_steps: The maximum number of steps per trajectory.
        template: A template for the agent to use for formatting prompts.
        bsize: Optional batch size override (if none, uses environment batch size)
    
    Returns:
        A list of trajectories.
    """
    # Use provided batch size or environment's batch size
    bsize = bsize or env.bsize
    all_trajectories = []
    
    # Calculate optimal batch size - use a smaller batch size to avoid OOM
    optimal_batch_size = min(bsize, 16)  # Start with a smaller max batch size to avoid OOM
    
    # Process in smaller batches to prevent memory issues
    num_batches = (num_trajectories + optimal_batch_size - 1) // optimal_batch_size  # Ceiling division
    
    # Set up progress tracking
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
        if hasattr(env, 'get_current_histories'):
            batch_obs = env.get_current_histories()
        
        # Get actions from agent
        actions = agent.get_action(batch_obs)
        
        # Take step in environment
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

        pbar.update(len(trajectories))
        accuracy = correct_total / trajectories_collected if trajectories_collected else 0
        pbar.set_postfix({
            "Reward": f"{accuracy:.4f}",  
            "Overall": f"{correct_total}/{trajectories_collected}"
        })
        
        del batch_obs, actions, batch_results, trajectories
    
    pbar.close()
    return all_trajectories




