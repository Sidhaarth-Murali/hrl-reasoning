#!/usr/bin/env bash
set -e

CONFIGS=(
  fullgrad_coef0.1
  fullgrad_coef0.01
  fullgrad_coef0.001
  stopgrad_coef0.1
  stopgrad_coef0.01
  stopgrad_coef0.001
)

GPUS=(0 1 2 3 4 5)
mkdir -p logs

# Launch each experiment on its own GPU
for i in "${!CONFIGS[@]}"; do
  CFG="${CONFIGS[$i]}"
  GPU="${GPUS[$i]}"
  echo "Starting $CFG on GPU $GPU…"
  
  CUDA_VISIBLE_DEVICES="$GPU" python run_bilevel_score.py \
    --config-name "$CFG" \
    > "logs/${CFG}.log" 2>&1 &

done

wait
echo "All runs complete."
