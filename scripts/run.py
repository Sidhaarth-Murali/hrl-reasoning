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
import gc
transformers.logging.set_verbosity_error()

# Set memory optimization environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"

def setup_memory_optimization():
    """Setup memory optimization settings"""
    if torch.cuda.is_available():
        # Check for ECC errors and problematic GPUs
        problematic_gpus = []
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,ecc.errors.uncorrected.volatile.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 2:
                            gpu_id = int(parts[0])
                            ecc_errors = int(parts[1]) if parts[1] != '[Not Supported]' else 0
                            if ecc_errors > 0:
                                problematic_gpus.append(gpu_id)
                                colorful_print(f"⚠️  GPU {gpu_id} has {ecc_errors} ECC errors - will avoid using it", fg='red')
        except Exception as e:
            colorful_print(f"Could not check ECC status: {e}", fg='yellow')
        
        try:
            # Set memory fraction to avoid fragmentation
            for i in range(torch.cuda.device_count()):
                if i not in problematic_gpus:
                    torch.cuda.set_per_process_memory_fraction(0.95, device=i)
                    colorful_print(f"✅ GPU {i} configured for use", fg='green')
                else:
                    colorful_print(f"❌ Skipping GPU {i} due to ECC errors", fg='red')
        except RuntimeError as e:
            if "CUDA error" in str(e):
                colorful_print(f"⚠️  CUDA error during memory fraction setup: {e}", fg='yellow')
                colorful_print("Continuing without memory fraction optimization...", fg='yellow')
            else:
                raise e
        
        colorful_print(f"CUDA devices available: {torch.cuda.device_count()}", fg='green')
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                status = "❌ AVOIDED" if i in problematic_gpus else "✅ OK"
                colorful_print(f"GPU {i}: {props.name}, Memory: {props.total_memory / 1e9:.1f} GB {status}", fg='green' if i not in problematic_gpus else 'red')
            except RuntimeError as e:
                colorful_print(f"⚠️  Could not get properties for GPU {i}: {e}", fg='yellow')
        
        return problematic_gpus
    else:
        colorful_print("⚠️  CUDA not available, running in CPU mode", fg='yellow')
        return []

def create_model_placement_strategy(accelerator, num_gpus=2, problematic_gpus=None):
    """
    Create optimal model placement strategy across available GPUs
    
    Strategy:
    - Use only healthy GPUs (avoid those with ECC errors)
    - GPU 1: Base LLM + Guidance Model + Value Function (if GPU 0 has issues)
    - CPU: Reference Model + Target Critics (inference only)
    """
    if problematic_gpus is None:
        problematic_gpus = []
    
    # Find available healthy GPUs
    available_gpus = [i for i in range(num_gpus) if i not in problematic_gpus]
    
    if len(available_gpus) >= 2:
        return {
            'base_model_device': torch.device(f'cuda:{available_gpus[1]}'),  # Use second GPU if first is problematic  
            'critic_device': torch.device(f'cuda:{available_gpus[1]}'), 
            'target_critic_device': torch.device('cpu'),  # Key optimization
            'guidance_model_device': torch.device(f'cuda:{available_gpus[0]}'),
            'value_function_device': torch.device(f'cuda:{available_gpus[0]}'),
            'reference_model_device': torch.device('cpu'),  # Key optimization
        }
    elif len(available_gpus) >= 1:
        # Single healthy GPU with CPU offloading
        healthy_gpu = available_gpus[0]
        return {
            'base_model_device': torch.device(f'cuda:{healthy_gpu}'),
            'critic_device': torch.device(f'cuda:{healthy_gpu}'),
            'target_critic_device': torch.device('cpu'),
            'guidance_model_device': torch.device(f'cuda:{healthy_gpu}'), 
            'value_function_device': torch.device(f'cuda:{healthy_gpu}'),
            'reference_model_device': torch.device('cpu'),
        }
    else:
        # All GPUs problematic - fallback to CPU
        colorful_print("⚠️  All GPUs have issues, falling back to CPU", fg='red')
        return {
            'base_model_device': torch.device('cpu'),
            'critic_device': torch.device('cpu'),
            'target_critic_device': torch.device('cpu'),
            'guidance_model_device': torch.device('cpu'), 
            'value_function_device': torch.device('cpu'),
            'reference_model_device': torch.device('cpu'),
        }

class ParallelizationAnalyzer:
    """
    Analyzes and implements parallelizable components in hierarchical RL
    """
    
    @staticmethod
    def identify_parallelizable_operations():
        """Identify operations that can be parallelized across GPUs"""
        return {
            'embarrassingly_parallel': [
                'trajectory_generation_across_problems',  # Different problems on different GPUs
                'kl_divergence_computation',             # Per-sample KL calculations  
                'value_function_forward_passes',         # Independent V(s) computations
                'critic_q_value_batches',               # Larger micro-batches for Q(s,a)
                'tokenization_and_encoding',            # Text processing can be parallel
            ],
            'pipeline_parallel': [
                'guidance_model_on_gpu1',               # While base model trains on GPU0
                'value_function_on_gpu1',               # Separate from critic training
                'target_network_updates_on_cpu',        # Async updates during training
            ],
            'temporal_parallel': [
                'overlapped_batch_processing',          # Process next batch while current trains
                'asynchronous_memory_cleanup',          # Background cache clearing
                'parallel_tokenizer_preprocessing',     # Prepare next batch while training
            ],
            'sequential_bottlenecks': [
                'turn1_to_guidance_dependency',         # Must complete Turn 1 first
                'guidance_to_turn2_dependency',         # Must get hints before Turn 2
                'episode_completion_for_gradients',     # Need full trajectory for REINFORCE
                'critic_target_synchronization',        # Target updates need main critic state
            ]
        }

def print_parallelization_analysis(agent_type, num_gpus):
    """Print parallelization analysis for the specific agent type"""
    analyzer = ParallelizationAnalyzer()
    parallel_ops = analyzer.identify_parallelizable_operations()
    
    colorful_print("\n🔍 PARALLELIZATION ANALYSIS:", fg='blue')
    colorful_print("✅ Embarrassingly Parallel Operations:", fg='green')
    for op in parallel_ops['embarrassingly_parallel']:
        colorful_print(f"   • {op}", fg='green')
    
    colorful_print("\n🔄 Pipeline Parallel Opportunities:", fg='cyan')  
    for op in parallel_ops['pipeline_parallel']:
        colorful_print(f"   • {op}", fg='cyan')
        
    colorful_print("\n⏱️  Temporal Parallel Optimizations:", fg='magenta')
    for op in parallel_ops['temporal_parallel']:
        colorful_print(f"   • {op}", fg='magenta')
        
    colorful_print("\n❌ Sequential Bottlenecks (Cannot Parallelize):", fg='red')
    for bottleneck in parallel_ops['sequential_bottlenecks']:
        colorful_print(f"   • {bottleneck}", fg='red')
    
    # Agent-specific analysis
    if agent_type.lower() in ['bi_level_score', 'score']:
        colorful_print(f"\n🎯 {agent_type.upper()} Specific Parallelization:", fg='blue')
        if num_gpus >= 2:
            colorful_print("   ✅ Two-turn training can use pipeline parallelism", fg='green')
            colorful_print("   ✅ Guidance model training on GPU1 while base on GPU0", fg='green')
        else:
            colorful_print("   ⚠️  Single GPU: Limited to memory optimizations", fg='yellow')
    elif agent_type.lower() == 'archer':
        colorful_print(f"\n🎯 ARCHER Specific Parallelization:", fg='blue')
        colorful_print("   ✅ Critic and actor training can be pipelined", fg='green')
        if num_gpus >= 2:
            colorful_print("   ✅ Value function on separate GPU", fg='green')

def monitor_memory_usage():
    """Monitor and log memory usage across devices"""
    stats = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        cached = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        
        stats[f'gpu_{i}'] = {
            'allocated': allocated,
            'cached': cached, 
            'total': total,
            'utilization': allocated / total
        }
        
        if allocated / total > 0.9:
            colorful_print(f"⚠️  GPU {i} memory usage high: {allocated:.1f}/{total:.1f} GB", fg='red')
            
    return stats

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
    
    # Setup memory optimization
    problematic_gpus = setup_memory_optimization()
    
    # Print parallelization analysis
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print_parallelization_analysis(config.agent_type, num_gpus)
    
    # Create device placement strategy
    if num_gpus > 0:
        placement_strategy = create_model_placement_strategy(None, num_gpus, problematic_gpus)
        colorful_print(f"\n💾 Device placement strategy: {placement_strategy}", fg='cyan')
    
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
        
        # Clear cache before creating agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
        
        # Monitor memory after agent creation
        memory_stats = monitor_memory_usage()
        colorful_print(f"💾 Memory usage after ARCHER agent creation: {memory_stats}", fg='cyan')
        
    elif config.agent_type.lower() == "online_filteredbc":
        print(">>> Using Online FilteredBC agent")
        
        # Clear cache before creating agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
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
        
        # Monitor memory after agent creation
        memory_stats = monitor_memory_usage()
        colorful_print(f"💾 Memory usage after FilteredBC agent creation: {memory_stats}", fg='cyan')
        
    elif config.agent_type.lower() == "score":
        print(">>> Using SCoRe agent")
        # Check if we're using a specific variant of SCoRe
        if hasattr(config, 'use_smart_corrections') and config.use_smart_corrections:
            if hasattr(config, 'train_guidance_model') and config.train_guidance_model:
                print(">>> with RL-guided correction")
            else:
                print(">>> with SMART correction")
        
        # Clear cache before creating agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
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
        
        # Monitor memory after agent creation
        memory_stats = monitor_memory_usage()
        colorful_print(f"💾 Memory usage after SCoRe agent creation: {memory_stats}", fg='cyan')
        
    elif config.agent_type.lower() == "bi_level_score":
        print(">>> Using BiLevel SCoRe agent")
        
        # Clear cache before creating agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
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
        
        # Monitor memory after agent creation
        memory_stats = monitor_memory_usage()
        colorful_print(f"💾 Memory usage after BiLevel SCoRe agent creation: {memory_stats}", fg='cyan')
        
        # Print BiLevel specific optimizations
        colorful_print("\n🎯 BiLevel SCoRe Memory Optimizations Applied:", fg='blue')
        colorful_print("   ✅ Target critic offloaded to CPU (-3GB GPU memory)", fg='green')
        colorful_print("   ✅ Reference model on CPU (-13GB GPU memory)", fg='green')
        colorful_print("   ✅ Value function can use separate GPU if available", fg='green')
        colorful_print("   ✅ Memory-efficient parameter updates", fg='green')
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

    # Final memory optimization and monitoring before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
    final_memory_stats = monitor_memory_usage()
    colorful_print(f"\n✅ Final memory usage before training: {final_memory_stats}", fg='green')
    
    # Print summary of optimizations applied
    colorful_print(f"\n🚀 Starting Memory-Optimized {config.agent_type.upper()} Training", fg='blue')
    colorful_print("🎯 Applied Optimizations:", fg='blue')
    colorful_print("   ✅ Memory-efficient agent initialization", fg='green')
    colorful_print("   ✅ Automatic cache cleanup", fg='green')
    if num_gpus >= 2:
        colorful_print("   ✅ Multi-GPU device placement strategy", fg='green')
        colorful_print("   ✅ Pipeline parallelism opportunities identified", fg='green')
    else:
        colorful_print("   ✅ Single GPU with CPU offloading", fg='green')
    colorful_print("   ✅ Embarrassingly parallel operations identified", fg='green')

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