import os, sys
import torch
import transformers
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from archer.environment import LLMBatchedMathEnv
from archer.data import DummyDataset
from archer.algorithms.bc import train_loop, plain_bc_loss
from archer.utils import colorful_print
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import numpy as np 
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader
import json
import os
import hydra
import random
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run behavior cloning for math environment')
parser.add_argument('--config', type=str, default="bc_math", 
                    help='Configuration to use (default: bc_math)')
args, unknown = parser.parse_known_args()

CONFIG_NAME = args.config
@hydra.main(version_base=None, config_path="./config/", config_name=CONFIG_NAME)
def main(config: "DictConfig"):
    colorful_print(">>> Configuration file: "+CONFIG_NAME+"<<<", fg='blue')
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    try:
        from huggingface_hub import login
        login(token=config.huggingface_token)
    except:
        print(">>> Huggingface token not found.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load environment to generate training data
    env = LLMBatchedMathEnv(
        env_load_path=config.env_load_path,
        device=device,
        cache_dir=config.cache_dir,
        max_tokens=config.max_tokens
    )
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(config.policy_lm, cache_dir=config.cache_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.policy_lm, cache_dir=config.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lm_lr)
    
    # Generate training data by interacting with the environment
    print(">>> Generating training data...")
    num_problems = 50  # Number of math problems to use for training
    training_data = []
    
    for i in tqdm(range(num_problems)):
        # Reset the environment
        observations = env.reset(i % len(env.env_list[0].problems))
        problem = observations[0]  # Get the first problem
        
        # Generate the answer using the environment's model
        with torch.no_grad():
            encoder_ids = env.tokenizer(problem, return_tensors='pt').to(device)
            outputs = env.model.generate(
                input_ids=encoder_ids['input_ids'],
                attention_mask=encoder_ids['attention_mask'],
                max_new_tokens=config.max_tokens,
                do_sample=True,
                temperature=0.7
            )
            answer = env.tokenizer.decode(outputs[0][encoder_ids['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Add to training data
        training_data.append({
            'observation': problem,
            'action': answer
        })
    
    # Create dataloader
    dataloader = DataLoader(DummyDataset(training_data), batch_size=config.batch_size, shuffle=True)
    
    # Start training
    print(">>> Starting behavior cloning training...")
    if config.use_wandb:
        wandb.login(key=config.wandb_key)
        wandb.init(project=config.project_name, name=config.run_name, config=dict(config))
    
    # Train the model using the simple BC train loop
    train_loop(
        model=model,
        tokenizer=tokenizer,
        bc_dataloader=dataloader,
        optimizer=optimizer,
        iterations=config.iterations,
        grad_accum_steps=config.grad_accum_steps,
        save_path=config.save_path,
        use_wandb=config.use_wandb
    )
    
    print(">>> Training complete!")

if __name__ == "__main__":
    main()
