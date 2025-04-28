import os, sys
import torch
import transformers
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from archer.environment import LLMBatchedMathEnv
from archer.models import ArcherAgent, CHAIAgent
from archer.algorithms import offpolicy_train_loop
from archer.utils import colorful_print
import torch.nn as nn
import numpy as np 
import wandb
from omegaconf import DictConfig, OmegaConf
import os
import hydra
from accelerate import Accelerator
from datetime import timedelta
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
import argparse
transformers.logging.set_verbosity_error()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run RL training for math environment')
parser.add_argument('--config-name', type=str, default="archer_math", 
                    help='Configuration to use (archer_math, bc_math, or score_math)')
args, unknown = parser.parse_known_args()

CONFIG_NAME = args.config_name
@hydra.main(version_base=None, config_path="./config/", config_name=CONFIG_NAME)
def main(config: "DictConfig"):
    colorful_print(">>> Configuration file: "+CONFIG_NAME+"<<<", fg='blue')
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    try:
        from huggingface_hub import login
        login(token=config.huggingface_token)
    except:
        print(">>> Huggingface token not found.")

    accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(18000)))
    device = accelerator.device

    # load environment
    if config.env_name == "math":
        env = LLMBatchedMathEnv(
            env_load_path=config.env_load_path,
            device=device,
            cache_dir=config.cache_dir,
            max_tokens=config.max_tokens
        )
        eval_env = env
    else:
        raise NotImplementedError("Only math environment is supported.")

    decode_f = lambda x:x
    # load decision model
    if config.agent_type.lower() == "chai":
        print(">>> Using CHAI agent")
        agent = CHAIAgent(
            device=device, 
            accelerator=accelerator, 
            temperature=config.temperature, 
            do_sample=config.do_sample, 
            policy_lm=config.policy_lm, 
            critic_lm=config.critic_lm, 
            cache_dir=config.cache_dir,
            max_new_tokens=config.max_new_tokens, 
            eos_str=config.eos_str
        )
        # if use chai, do not update the actor
        config.warmup_iter = config.iterations
    elif config.agent_type.lower() == "archer":
        print(">>> Using ArCHer agent")
        agent = ArcherAgent(
            device=device,
            accelerator=accelerator, 
            temperature=config.temperature,
            do_sample=config.do_sample, 
            policy_lm=config.policy_lm,
            critic_lm=config.critic_lm,
            cache_dir=config.cache_dir,
            max_new_tokens=config.max_new_tokens,
            use_lora=config.use_lora,
            eos_str='\n'
        )
    elif config.agent_type.lower() == "online_filteredbc":
        print(">>> Using Online FilteredBC agent")
        # the agent is the same as ArCHer, only the trainer will be different
        agent = ArcherAgent(device=device, accelerator=accelerator, 
                            temperature=config.temperature, do_sample=config.do_sample, 
                            policy_lm=config.policy_lm, critic_lm=config.critic_lm,
                            cache_dir=config.cache_dir, max_new_tokens=config.max_new_tokens)
    elif config.agent_type.lower() == "score":
        print(">>> Using SCoRe agent")
        # SCoRe uses the same agent architecture as ArCHer, but different training algorithm
        agent = ArcherAgent(
            device=device,
            accelerator=accelerator, 
            temperature=config.temperature,
            do_sample=config.do_sample, 
            policy_lm=config.policy_lm,
            critic_lm=config.policy_lm,  # SCoRe doesn't use a separate critic model
            cache_dir=config.cache_dir,
            max_new_tokens=config.max_new_tokens,
            use_lora=config.use_lora,
            eos_str='\n'
        )
    else:
        raise NotImplementedError("Agent not implemented.")
    tokenizer = agent.tokenizer

    if config.checkpoint_path is not None:
        state_dict = torch.load(config.checkpoint_path, map_location=device)['model_state_dict']
        agent.model.load_state_dict(state_dict)
    agent = accelerator.prepare(agent)

    if config.use_wandb and accelerator.is_main_process:
        wandb.login(key=config.wandb_key)
        wandb.init(project=config.project_name, name=config.run_name, config=dict(config))

    offpolicy_train_loop(
        env=env,
        eval_env=eval_env,
        agent=agent,
        tokenizer=tokenizer,
        accelerator=accelerator,
        decode_f=decode_f,
        **config
    )

if __name__ == "__main__":
    main()