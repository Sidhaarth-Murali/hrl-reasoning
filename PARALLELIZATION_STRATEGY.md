# Hierarchical RL Parallelization Strategy & Memory Optimization

## Why This Process "Is Not Really Parallelizable Just Like That"

### 🔄 Sequential Dependencies in Training Flow

Unlike embarrassingly parallel tasks (like independent model inference), this hierarchical RL system has **fundamental sequential dependencies**:

```
Turn 1: Agent generates solution
   ↓
Guidance: Model analyzes Turn 1 → provides hints  
   ↓
Turn 2: Agent generates revised solution using hints
   ↓
Reward: Based on Turn 2 correctness
   ↓ 
Updates: All models trained on complete episodes
```

**Critical Constraints:**
- Guidance model **cannot** generate hints until Turn 1 is complete
- Turn 2 **cannot** begin until guidance is provided
- Policy gradients **require** complete episode trajectories
- Target network updates need synchronized critic states

This creates a **pipeline dependency** where each stage must complete before the next begins.

### 🧠 Memory Bottleneck Analysis

Your OOM error occurs during `ArcherAgent` initialization because all models are loaded simultaneously:

```python
# Memory allocation during initialization:
Base LLM (LLaMA 3.2):     ~7-13 GB
Main Critic (RoBERTa):    ~2-3 GB  
Target Critic (RoBERTa):  ~2-3 GB  [COPIED FROM MAIN]
Reference Model:          ~7-13 GB [COPIED FROM BASE]
Guidance Model:           ~7-13 GB [COPIED FROM BASE]
Value Function:           ~1-2 GB
─────────────────────────────────────
Total:                    ~26-47 GB ❌ > 44.53 GB available
```

## 🎯 Optimization Strategy: Hybrid CPU-GPU Distribution

### 1. **Smart Device Placement**

**GPU 0 (Primary Training):**
- Base LLM (active training)
- Main Critic (active training)

**GPU 1 (Secondary Training):**  
- Guidance Model (episodic training)
- Value Function (episodic training)

**CPU (Inference Only):**
- Reference Model (KL divergence calculation only)
- Target Critic (periodic updates only)

### 2. **Memory-Efficient Target Updates**

Instead of full GPU allocation for target critic updates:

```python
def _memory_efficient_target_update(self, tau):
    # Process parameters in chunks
    for chunk in parameter_chunks:
        source_data = source_param.data.cpu()  # Temporary CPU copy
        target_param.data.copy_(target_param.data * (1-tau) + source_data * tau)
        del source_data  # Immediate cleanup
        torch.cuda.empty_cache()
```

### 3. **Parallelizable Components**

#### ✅ **Embarrassingly Parallel Operations:**
- **Batch Generation**: Multiple problems can be processed independently
- **KL Divergence**: Each sample's KL can be computed separately  
- **Value Function Training**: Independent of other model updates
- **Critic Q/V Computation**: Can use larger micro-batches

#### ❌ **Sequential Bottlenecks:**
- Episode collection (Turn 1 → Guidance → Turn 2)
- Policy gradient calculation (needs complete trajectories)
- Advantage computation (critic-dependent)

### 4. **Temporal Batching Strategy**

```python
# Phase 1: Parallel trajectory generation (embarrassingly parallel)
gpu_0_trajectories = generate_trajectories(problems[0:256], gpu=0)
gpu_1_trajectories = generate_trajectories(problems[256:512], gpu=1)

# Phase 2: Sequential training (pipeline dependencies)
for batch in collected_trajectories:
    train_critic(batch)      # GPU 0
    train_actor(batch)       # GPU 0  
    train_guidance(batch)    # GPU 1
    train_value_func(batch)  # GPU 1
```

## 🔧 Implementation Optimizations

### Memory Optimizations Applied:

1. **Target Critic CPU Offloading**
   ```python
   self.target_critic = DoubleCritic(cpu_device, ...)  # CPU placement
   ```

2. **Chunked Parameter Updates**
   ```python
   # Process 10 parameters at a time instead of all at once
   for chunk in chunks(parameters, size=10):
       update_chunk_on_cpu(chunk)
   ```

3. **Smart Tokenization**
   ```python
   # Only move to device if not already there
   if obs_ids['input_ids'].device != self.device:
       obs_ids = {k: v.to(self.device) for k, v in obs_ids.items()}
   ```

4. **Immediate Memory Cleanup**
   ```python
   del intermediate_tensors
   torch.cuda.empty_cache()
   ```

### Batch Size Optimization:

```python
def optimize_batch_processing(available_memory_gb):
    if available_memory_gb > 40:
        return {'base_batch': 16, 'critic_batch': 32, 'micro_batch': 8}
    elif available_memory_gb > 20:  
        return {'base_batch': 8, 'critic_batch': 16, 'micro_batch': 4}
    else:
        return {'base_batch': 4, 'critic_batch': 8, 'micro_batch': 2}
```

## 🚀 Expected Performance Improvements

### Memory Savings:
- **Target Critic**: 2-3 GB saved (moved to CPU)
- **Reference Model**: 7-13 GB saved (moved to CPU)  
- **Chunked Updates**: ~50% reduction in peak memory during updates
- **Total Savings**: ~15-20 GB, bringing usage to ~30-35 GB ✅

### Parallelization Gains:
- **Trajectory Generation**: 2x speedup with 2 GPUs
- **Independent Model Updates**: Can overlap critic/guidance training
- **Batch Processing**: More efficient micro-batching

### Trade-offs:
- **CPU-GPU Transfer**: Small overhead for target critic operations
- **Pipeline Complexity**: More complex device management
- **Memory Monitoring**: Need careful cache management

## 🔧 Usage

Run the memory-optimized training:

```bash
python scripts/run_memory_optimized.py
```

This implements:
- Automatic device placement strategy
- Memory monitoring and optimization
- Smart batch size selection based on available memory
- Hybrid CPU-GPU model distribution

The key insight is that hierarchical RL training has **unavoidable sequential dependencies** in the episode collection and training pipeline, but we can still **parallelize independent operations** and **optimize memory usage** through smart device placement and efficient parameter management. 