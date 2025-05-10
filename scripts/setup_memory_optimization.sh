#!/bin/bash

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Enable memory efficient attention
export TRANSFORMERS_USE_MEMORY_EFFICIENT_ATTENTION=1

# Enable gradient checkpointing
export TRANSFORMERS_USE_GRADIENT_CHECKPOINTING=1

# Enable mixed precision training
export TRANSFORMERS_USE_BF16=1

# Set PyTorch memory management
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set environment variables for better memory management
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8
export PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:4

# Print current settings
echo "Memory optimization settings applied:"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "CUBLAS_WORKSPACE_CONFIG: $CUBLAS_WORKSPACE_CONFIG"
echo "TRANSFORMERS_USE_MEMORY_EFFICIENT_ATTENTION: $TRANSFORMERS_USE_MEMORY_EFFICIENT_ATTENTION"
echo "TRANSFORMERS_USE_GRADIENT_CHECKPOINTING: $TRANSFORMERS_USE_GRADIENT_CHECKPOINTING"
echo "TRANSFORMERS_USE_BF16: $TRANSFORMERS_USE_BF16" 