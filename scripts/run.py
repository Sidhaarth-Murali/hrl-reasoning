import os, sys
import torch
import transformers
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from archer.environment import LLMBatchedMathEnv, LLMBatchedCodeEnv
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
parser = argparse.ArgumentParser(description='Run RL training for math or code environment')
parser.add_argument('--config-name', type=str, default="archer_math", 
                    help='Configuration to use (archer_math, bc_math, score_math, etc.)')
parser.add_argument('--env-type', type=str, default="math",
                    help='Environment type (math or code)')
args = parser.parse_args()

ENV_TYPE = args.env_type
CONFIG_NAME = args.config_name

# Select the appropriate config directory based on environment type
if ENV_TYPE == "math":
    config_directory = "config/math_configs/"
elif ENV_TYPE == "code":
    config_directory = "config/code_configs/"
else:
    config_directory = "config/"

# Modified to use sys.argv[1:0] to prevent Hydra from seeing our parsed arguments
@hydra.main(version_base=None, config_path=config_directory, config_name=CONFIG_NAME)
def main(config: "DictConfig"):
    colorful_print(f">>> Configuration file: {CONFIG_NAME} for {ENV_TYPE} environment <<<", fg='blue')
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    try:
        from huggingface_hub import login  
        login(token=config.huggingface_token)
    except:
        print(">>> Huggingface token not found.")

    accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(18000)))
    device = accelerator.device

    # Determine environment type from args (override config if present)
    env_type = ENV_TYPE if ENV_TYPE else (config.env_type if hasattr(config, 'env_type') else "math")
    
    # Common environment kwargs
    env_kwargs = {
        "env_load_path": config.env_load_path,
        "device": device,
        "cache_dir": config.cache_dir,
        "max_tokens": config.max_tokens,
        "bsize": config.rollout_size,
    }
    
    # Add SCoRe specific parameters if they exist in config
    if hasattr(config, 'use_smart_corrections'):
        env_kwargs["use_smart_corrections"] = config.use_smart_corrections
        
    if hasattr(config, 'correction_model_path') and config.correction_model_path:
        env_kwargs["correction_model_path"] = config.correction_model_path
        
    if hasattr(config, 'train_guidance_model'):
        env_kwargs["train_guidance_model"] = config.train_guidance_model
    
    # Add model_name 
    if hasattr(config, 'policy_lm'):
        env_kwargs["model_name"] = config.policy_lm
        
    # load environment
    if env_type == "code":
        # Add language parameter for code environment
        if hasattr(config, 'language'):
            env_kwargs["language"] = config.language
        
        # Add data_path parameter
        if hasattr(config, 'data_path'):
            env_kwargs["data_path"] = config.data_path
            
        colorful_print(f">>> Using CodeEnv with data from {env_kwargs.get('data_path', 'default dataset')}", fg='green')
        env = LLMBatchedCodeEnv(**env_kwargs)
        
        # Create a separate eval environment with smaller batch size
        eval_env_kwargs = env_kwargs.copy()
        eval_env_kwargs["bsize"] = config.eval_size
        eval_env = LLMBatchedCodeEnv(**eval_env_kwargs)
    elif env_type == "math":
        # Add data_path parameter for math environment
        if hasattr(config, 'data_path'):
            env_kwargs["data_path"] = config.data_path
            
        colorful_print(f">>> Using MathEnv with data from {env_kwargs.get('data_path', 'default dataset')}", fg='green')
        env = LLMBatchedMathEnv(**env_kwargs)
        
        # Create a separate eval environment with smaller batch size
        eval_env_kwargs = env_kwargs.copy()
        eval_env_kwargs["bsize"] = config.eval_size
        eval_env = LLMBatchedMathEnv(**eval_env_kwargs)
    else:
        raise NotImplementedError(f"Environment {env_type} not implemented.")

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
            use_gradient_checkpointing=config.use_gradient_checkpointing if hasattr(config, 'use_gradient_checkpointing') else False,
            use_memory_efficient_attention=config.use_memory_efficient_attention if hasattr(config, 'use_memory_efficient_attention') else False,
            load_in_8bit=config.load_in_8bit if hasattr(config, 'load_in_8bit') else False,
            eos_str='\n'
        )
    elif config.agent_type.lower() == "online_filteredbc":
        print(">>> Using Online FilteredBC agent")
        # the agent is the same as ArCHer, only the trainer will be different
        agent = ArcherAgent(
            device=device, 
            accelerator=accelerator,
            temperature=config.temperature, 
            do_sample=config.do_sample, 
            policy_lm=config.policy_lm, 
            critic_lm=config.critic_lm,
            cache_dir=config.cache_dir, 
            max_new_tokens=config.max_new_tokens,
            use_gradient_checkpointing=config.use_gradient_checkpointing if hasattr(config, 'use_gradient_checkpointing') else False,
            use_memory_efficient_attention=config.use_memory_efficient_attention if hasattr(config, 'use_memory_efficient_attention') else False,
            load_in_8bit=config.load_in_8bit if hasattr(config, 'load_in_8bit') else False
        )
    elif config.agent_type.lower() == "score":
        print(">>> Using SCoRe agent")
        # Check if we're using a specific variant of SCoRe
        if hasattr(config, 'use_smart_corrections') and config.use_smart_corrections:
            if hasattr(config, 'train_guidance_model') and config.train_guidance_model:
                print(">>> with RL-guided correction")
            else:
                print(">>> with SMART correction")
                
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
            use_gradient_checkpointing=config.use_gradient_checkpointing if hasattr(config, 'use_gradient_checkpointing') else False,
            use_memory_efficient_attention=config.use_memory_efficient_attention if hasattr(config, 'use_memory_efficient_attention') else False,
            load_in_8bit=config.load_in_8bit if hasattr(config, 'load_in_8bit') else False,
            eos_str='\n'
        )
    elif config.agent_type.lower() == "bi_level_score":
        print(">>> Using BiLevel SCoRe agent")
        # BiLevel SCoRe agent
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
            use_gradient_checkpointing=config.use_gradient_checkpointing if hasattr(config, 'use_gradient_checkpointing') else False,
            use_memory_efficient_attention=config.use_memory_efficient_attention if hasattr(config, 'use_memory_efficient_attention') else False,
            load_in_8bit=config.load_in_8bit if hasattr(config, 'load_in_8bit') else False,
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
    # Clear command line arguments for Hydra
    sys.argv = [sys.argv[0]]
    main()