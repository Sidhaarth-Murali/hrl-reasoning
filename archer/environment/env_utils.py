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



from tqdm import tqdm

def batch_interact_environment(agent, tokenizer, env, num_trajectories, 
        post_f=lambda x: x, use_tqdm=True, decode_f=lambda x: x, env_idx=None):
    """
    Batched environment interaction to collect trajectories.
    """
    bsize = env.bsize
    all_trajectories = []    

    for _ in tqdm(range(num_trajectories // bsize), desc="Batches", disable=not use_tqdm, position=1):
        trajectories = [[] for _ in range(bsize)]
        batch_obs = env.reset(idx=env_idx)
        batch_done = [False] * bsize

        for _ in tqdm(range(256), desc="Steps", disable=not use_tqdm, position=2):
            if all(batch_done):
                break  

            action = agent.get_action(batch_obs)
            batch_return = env.step(decode_f(action))

            for i, result in zip(range(bsize), batch_return):
                if result is None:
                    continue
                next_obs, r, done = result

                trajectories[i].append({
                    "observation": batch_obs[i],
                    "next_observation": next_obs,
                    "reward": r,
                    "done": done,
                    "action": action[i]
                })
                batch_obs[i] = next_obs
                batch_done[i] = done

        all_trajectories += [post_f(add_mc_return(add_trajectory_reward(trajectory))) 
                              for trajectory in trajectories]

    return all_trajectories



