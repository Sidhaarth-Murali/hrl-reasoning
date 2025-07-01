#!/bin/bash
#
# Advanced Distributed BiLevel SCoRe Training Launcher
# 
# This script launches distributed training with optimal configuration
# for bilevel SCoRe with full gradient flow (fullgrad_coef0.01)
#
# Usage:
#   ./launch_distributed_bilevel.sh [num_gpus] [config_name]
#
# Examples:
#   ./launch_distributed_bilevel.sh 2 fullgrad_coef0.01_distributed
#   ./launch_distributed_bilevel.sh 4 fullgrad_coef0.01_distributed
#

set -e  # Exit on any error

# Configuration
NUM_GPUS=${1:-2}
CONFIG_NAME=${2:-"fullgrad_coef0.01_distributed"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Verify GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi || { echo "❌ NVIDIA GPUs not found"; exit 1; }

AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "✅ Found $AVAILABLE_GPUS GPU(s)"

if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "⚠️  Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    echo "📝 Using $AVAILABLE_GPUS GPUs instead"
    NUM_GPUS=$AVAILABLE_GPUS
fi

# Set optimal environment variables
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.6"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# Configure NCCL for optimal performance
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
export NCCL_P2P_DISABLE=0  # Enable P2P if supported
export NCCL_SHM_DISABLE=0  # Enable shared memory

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF,max_split_size_mb:512"

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$ROOT_DIR/outputs/distributed_${CONFIG_NAME}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "📂 Output directory: $OUTPUT_DIR"
echo "🚀 Launching distributed training with $NUM_GPUS GPUs..."
echo "⚙️  Configuration: $CONFIG_NAME"

# Log system information
echo "💻 System Information:" | tee "$OUTPUT_DIR/system_info.log"
echo "  - Hostname: $(hostname)" | tee -a "$OUTPUT_DIR/system_info.log"
echo "  - Date: $(date)" | tee -a "$OUTPUT_DIR/system_info.log"
echo "  - Python: $(python --version)" | tee -a "$OUTPUT_DIR/system_info.log"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')" | tee -a "$OUTPUT_DIR/system_info.log"
echo "  - CUDA: $(python -c 'import torch; print(torch.version.cuda)')" | tee -a "$OUTPUT_DIR/system_info.log"
echo "  - GPUs: $NUM_GPUS" | tee -a "$OUTPUT_DIR/system_info.log"

# GPU information
echo "🔧 GPU Information:" | tee -a "$OUTPUT_DIR/system_info.log"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | tee -a "$OUTPUT_DIR/system_info.log"

# Check for ECC errors
echo "🔍 Checking for GPU ECC errors..." | tee -a "$OUTPUT_DIR/system_info.log"
nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits | tee -a "$OUTPUT_DIR/system_info.log"

# Function to cleanup on exit
cleanup() {
    echo "🧹 Cleaning up processes..."
    pkill -f "run_distributed_bilevel.py" || true
    sleep 2
}
trap cleanup EXIT

# Launch distributed training with accelerate
echo "🎯 Starting distributed training..."

cd "$SCRIPT_DIR"

# Option 1: Using accelerate launch (recommended)
if command -v accelerate &> /dev/null; then
    echo "📊 Using accelerate launch..."
    
    # Generate accelerate config if it doesn't exist
    ACCELERATE_CONFIG="config/accelerate_config/distributed_bilevel.yaml"
    if [ ! -f "$ACCELERATE_CONFIG" ]; then
        echo "⚠️  Accelerate config not found, using default multi-GPU setup"
        accelerate config default --config_file "$ACCELERATE_CONFIG"
    fi
    
    accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        --num_processes "$NUM_GPUS" \
        --mixed_precision bf16 \
        --dynamo_backend no \
        run_distributed_bilevel.py \
        --config-name "$CONFIG_NAME" \
        2>&1 | tee "$OUTPUT_DIR/training.log"

# Option 2: Using torchrun (fallback)
else
    echo "📊 Using torchrun..."
    
    torchrun \
        --standalone \
        --nproc_per_node="$NUM_GPUS" \
        --nnodes=1 \
        --max_restarts=1 \
        run_distributed_bilevel.py \
        --config-name "$CONFIG_NAME" \
        2>&1 | tee "$OUTPUT_DIR/training.log"
fi

# Check training success
if [ $? -eq 0 ]; then
    echo "🎉 Distributed training completed successfully!"
    echo "📂 Results saved to: $OUTPUT_DIR"
    
    # Save final system stats
    echo "📊 Final system statistics:" | tee "$OUTPUT_DIR/final_stats.log"
    nvidia-smi | tee -a "$OUTPUT_DIR/final_stats.log"
    
    # Archive logs
    echo "📦 Archiving logs..."
    tar -czf "$OUTPUT_DIR/logs_archive.tar.gz" -C "$OUTPUT_DIR" . 2>/dev/null || true
    
else
    echo "❌ Training failed with exit code $?"
    echo "📝 Check logs in: $OUTPUT_DIR/training.log"
    exit 1
fi

# Performance summary
echo ""
echo "📈 Training Summary:"
echo "  - Configuration: $CONFIG_NAME"
echo "  - GPUs used: $NUM_GPUS"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - Log file: $OUTPUT_DIR/training.log"
echo ""
echo "🚀 To monitor training progress:"
echo "  tail -f $OUTPUT_DIR/training.log"
echo ""
echo "🔍 To check GPU utilization:"
echo "  watch -n 1 nvidia-smi" 