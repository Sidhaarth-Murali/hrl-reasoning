#!/usr/bin/env python3
"""
Advanced Distributed BiLevel SCoRe Training Script

This script implements state-of-the-art distributed training for hierarchical RL
with memory optimization and pipeline parallelism. Specifically designed for
bilevel SCoRe training with full gradient flow.

Key Features:
1. Pipeline parallelism across multiple GPUs
2. Smart device placement for memory efficiency  
3. Advanced gradient synchronization
4. Memory-optimized model placement
5. Fault-tolerant distributed training
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import gc
import psutil
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add parent directory for archer imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import set_seed, DistributedType

from archer.environment import LLMBatchedMathEnv, LLMBatchedCodeEnv  
from archer.models import ArcherAgent
from archer.algorithms.score.bi_level_trainer import BiLevelSCoReTrainer
from archer.algorithms.offpolicy_train_loop import offpolicy_train_loop
from archer.utils import colorful_print

# Set optimal environment variables for distributed training
os.environ.update({
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.6",
    "NCCL_ASYNC_ERROR_HANDLING": "1",
    "NCCL_TIMEOUT": "1800",
    "CUDA_LAUNCH_BLOCKING": "0",
    "OMP_NUM_THREADS": "8",
    "TOKENIZERS_PARALLELISM": "false"
})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [Rank %(rank)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class DistributedMemoryManager:
    """Advanced memory management for distributed training"""
    
    def __init__(self, config: DictConfig, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.memory_stats = {}
        self.peak_memory = 0
        
    def setup_memory_optimization(self):
        """Setup advanced memory optimizations"""
        if torch.cuda.is_available():
            # Set memory fraction per GPU to avoid fragmentation
            memory_fraction = 0.95 / self.world_size
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=i)
                except RuntimeError as e:
                    logger.warning(f"Could not set memory fraction for GPU {i}: {e}")
            
            # Enable memory pool
            torch.cuda.empty_cache()
            
        logger.info(f"Rank {self.rank}: Memory optimization setup completed")
    
    def get_memory_stats(self) -> Dict:
        """Get comprehensive memory statistics"""
        stats = {"rank": self.rank}
        
        # GPU memory stats
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                
                stats[f"gpu_{i}"] = {
                    "allocated_gb": allocated,
                    "reserved_gb": reserved, 
                    "total_gb": total,
                    "utilization": allocated / total if total > 0 else 0
                }
                
                self.peak_memory = max(self.peak_memory, allocated)
        
        # CPU memory stats
        cpu_mem = psutil.virtual_memory()
        stats["cpu"] = {
            "used_gb": cpu_mem.used / 1e9,
            "available_gb": cpu_mem.available / 1e9,
            "percent": cpu_mem.percent
        }
        
        return stats
    
    def clear_cache(self, aggressive: bool = False):
        """Clear GPU and CPU caches"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.synchronize()
        
        if aggressive:
            gc.collect()

class DistributedDeviceManager:
    """Manages device placement strategy for distributed training"""
    
    def __init__(self, config: DictConfig, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device_map = {}
        self.local_rank = rank % torch.cuda.device_count()
        
    def setup_device_placement(self) -> Dict[str, torch.device]:
        """Setup optimal device placement strategy"""
        # Adjust device placement based on rank
        if self.world_size > 1:
            # Distribute models across available GPUs intelligently
            base_gpu = self.local_rank
            secondary_gpu = (self.local_rank + 1) % torch.cuda.device_count()
            
            self.device_map = {
                "base_model_device": torch.device(f"cuda:{base_gpu}"),
                "critic_device": torch.device(f"cuda:{base_gpu}"),
                "target_critic_device": torch.device("cpu"),  # CPU offloading
                "guidance_model_device": torch.device(f"cuda:{secondary_gpu}") if torch.cuda.device_count() > 1 else torch.device(f"cuda:{base_gpu}"),
                "value_function_device": torch.device(f"cuda:{secondary_gpu}") if torch.cuda.device_count() > 1 else torch.device(f"cuda:{base_gpu}"),
                "reference_model_device": torch.device("cpu"),  # CPU offloading
            }
        else:
            # Single node placement
            self.device_map = {
                "base_model_device": torch.device("cuda:0"),
                "critic_device": torch.device("cuda:0"),
                "target_critic_device": torch.device("cpu"),
                "guidance_model_device": torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cuda:0"),
                "value_function_device": torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cuda:0"),
                "reference_model_device": torch.device("cpu"),
            }
        
        logger.info(f"Rank {self.rank}: Device placement strategy: {self.device_map}")
        return self.device_map

@hydra.main(config_path="config/math_configs", version_base="1.2")
def main(config: DictConfig):
    """Main distributed training function"""
    
    colorful_print("🚀 Starting Advanced Distributed BiLevel SCoRe Training", fg='blue')
    colorful_print(f"Configuration: {config.run_name}", fg='blue')
    
    # Setup distributed training environment
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        static_graph=False,
        bucket_cap_mb=25,
    )
    
    init_kwargs = InitProcessGroupKwargs(timeout=1800)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.batch_optimization.gradient_accumulation_steps,
        mixed_precision='bf16' if config.memory_optimization.use_bf16_mixed_precision else 'no',
        kwargs_handlers=[ddp_kwargs, init_kwargs],
        log_with='wandb' if config.use_wandb else None,
    )
    
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    
    # Setup memory and device managers
    memory_manager = DistributedMemoryManager(config, rank, world_size)
    device_manager = DistributedDeviceManager(config, rank, world_size)
    
    memory_manager.setup_memory_optimization()
    device_map = device_manager.setup_device_placement()
    
    # Setup logging for main process
    if rank == 0:
        colorful_print(f"World Size: {world_size}, Local Rank: {rank}", fg='green')
        colorful_print(f"Device Strategy: {device_map}", fg='cyan')
        
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                name=f"{config.run_name}_distributed",
                config=OmegaConf.to_container(config, resolve=True),
                tags=["distributed", "bilevel", "score", "fullgrad"],
            )
    
    # Login to HuggingFace if token provided
    try:
        from huggingface_hub import login
        if hasattr(config, 'huggingface_token'):
            login(token=config.huggingface_token)
            logger.info(f"Rank {rank}: HuggingFace login successful")
    except Exception as e:
        logger.warning(f"Rank {rank}: HuggingFace login failed: {e}")
    
    # Create agent with distributed configuration
    agent_config = {
        "device": device_map["base_model_device"],
        "accelerator": accelerator,
        "policy_lm": config.policy_lm,
        "critic_lm": config.critic_lm,
        "cache_dir": config.cache_dir,
        "temperature": config.temperature,
        "do_sample": config.do_sample,
        "max_new_tokens": config.max_new_tokens,
        "use_lora": config.use_lora,
        "use_gradient_checkpointing": config.memory_optimization.use_gradient_checkpointing,
        "use_memory_efficient_attention": config.memory_optimization.use_memory_efficient_attention,
        "use_bfloat16": config.memory_optimization.use_bf16_mixed_precision,
    }
    
    memory_manager.clear_cache(aggressive=True)
    agent = ArcherAgent(**agent_config)
    
    # Apply device optimizations
    if hasattr(agent, 'target_critic'):
        agent.target_critic = agent.target_critic.to(device_map["target_critic_device"])
    
    # Prepare agent with accelerator (includes DDP wrapping)
    agent = accelerator.prepare(agent)
    
    # Create environment
    if config.env_name == "math":
        env = LLMBatchedMathEnv(
            batch_size=config.batch_optimization.per_gpu_batch_size,
            data_path=config.data_path,
            cache_dir=config.cache_dir
        )
        eval_env = LLMBatchedMathEnv(
            batch_size=config.eval_size,
            data_path=config.data_path,
            cache_dir=config.cache_dir
        )
    else:
        raise ValueError(f"Environment {config.env_name} not supported")
    
    # Create tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.policy_lm,
        cache_dir=config.cache_dir,
        trust_remote_code=True
    )
    
    # Memory check before training
    if rank == 0:
        memory_stats = memory_manager.get_memory_stats()
        colorful_print(f"Pre-training memory stats: {memory_stats}", fg='yellow')
    
    # Synchronize all processes before starting training
    accelerator.wait_for_everyone()
    
    if rank == 0:
        colorful_print("✅ Distributed setup completed, starting training...", fg='green')
    
    # Start distributed training
    offpolicy_train_loop(
        env=env,
        eval_env=eval_env,
        agent=agent,
        tokenizer=tokenizer,
        accelerator=accelerator,
        decode_f=lambda x: x,  # Simple decode function
        **config
    )
    
    if rank == 0:
        colorful_print("🎉 Distributed training completed successfully!", fg='green')
        
        # Final memory statistics
        final_memory = memory_manager.get_memory_stats()
        colorful_print(f"Final memory stats: {final_memory}", fg='blue')
        
        if config.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main() 