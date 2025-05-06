"""
RL-guided SCoRe training script.
This script runs the RL-guided SCoRe training process using the config from hydra.
"""

import os
import torch
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

from archer.environment import LLMBatchedMathEnv
from archer.algorithms.offpolicy_train_loop import offpolicy_train_loop
from archer.agent import Agent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@hydra.main(config_path="scripts/config", config_name="rl_guided_score_math", version_base="1.2")
def main(cfg: DictConfig):
    """Main entry point for RL-guided SCoRe implementation."""
    
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set up accelerator
    accelerator = Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=cfg.grad_accum_steps,
        log_with="wandb" if cfg.use_wandb else None,
    )
    
    # Set up agent with model
    device = accelerator.device
    logger.info(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.policy_lm,
        cache_dir=cfg.cache_dir,
        token=cfg.huggingface_token,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.policy_lm,
        cache_dir=cfg.cache_dir,
        token=cfg.huggingface_token,
    )
    
    # Initialize agent
    agent = Agent(model=model, max_tokens=cfg.max_tokens)
    agent.model = accelerator.prepare(agent.model)
    
    # Create env
    env = LLMBatchedMathEnv(
        env_load_path=cfg.env_load_path,
        cache_dir=cfg.cache_dir,
        device=device,
        max_tokens=cfg.max_tokens,
        bsize=cfg.rollout_size,
        data_path="dataset/MATH.csv",
        correction_model_path=cfg.correction_model_path,
        use_smart_corrections=cfg.use_smart_corrections,
        train_guidance_model=cfg.train_guidance_model,
    )
    
    # Create eval env (smaller batch size for evaluation)
    eval_env = LLMBatchedMathEnv(
        env_load_path=cfg.env_load_path,
        cache_dir=cfg.cache_dir,
        device=device,
        max_tokens=cfg.max_tokens,
        bsize=cfg.eval_size,
        data_path="dataset/MATH.csv",
        correction_model_path=cfg.correction_model_path,
        use_smart_corrections=cfg.use_smart_corrections,
        train_guidance_model=cfg.train_guidance_model,
    )
    
    logger.info("Starting RL-guided SCoRe training...")
    
    # Run training loop
    offpolicy_train_loop(
        env=env,
        eval_env=eval_env,
        agent=agent,
        tokenizer=tokenizer,
        accelerator=accelerator,
        grad_accum_steps=cfg.grad_accum_steps,
        warmup_iter=cfg.warmup_iter,
        rollout_size=cfg.rollout_size,
        eval_size=cfg.eval_size,
        batch_size=cfg.batch_size,
        capacity=cfg.capacity,
        iterations=cfg.iterations,
        env_idx=None,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        critic_lr=cfg.critic_lr,
        lm_lr=cfg.lm_lr,
        gamma=cfg.gamma,
        tau=cfg.tau,
        use_wandb=cfg.use_wandb,
        env_load_path=cfg.env_load_path,
        actor_epochs=cfg.actor_epochs,
        max_grad_norm=cfg.max_grad_norm,
        save_path=cfg.save_path,
        save_freq=cfg.save_freq,
        eval_freq=cfg.eval_freq,
        agent_type="score",
        # SCoRe specific parameters
        alpha=cfg.alpha,
        beta1=cfg.beta1,
        beta2=cfg.beta2,
        stage1_steps=cfg.stage1_steps,
        stage2_steps=cfg.stage2_steps,
        # SMART_SCoRe additions
        use_smart_corrections=cfg.use_smart_corrections,
        correction_model_path=cfg.correction_model_path,
        # RL-Guided SCoRe additions
        train_guidance_model=cfg.train_guidance_model,
        guidance_lr=cfg.guidance_lr,
        guidance_model_path=cfg.guidance_model_path,
        guidance_kl_coef=cfg.guidance_kl_coef,
    )
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 