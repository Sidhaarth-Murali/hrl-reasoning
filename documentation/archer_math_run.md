# Running ArCHer on the MATH Dataset

This document provides instructions for setting up and running the ArCHer reinforcement learning framework on the MATH dataset for solving mathematical problems step-by-step.

## Environment Setup

ArCHer requires a Python environment with specific dependencies. Follow these steps to set up your environment:

```bash
# Create a new conda environment
conda create -n archer-math python=3.10

# Activate the environment
conda activate archer-math

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install transformers datasets accelerate hydra-core omegaconf pandas wandb tqdm ipdb
```

## Dataset Preparation

The MATH dataset should be stored in CSV format with at least the following columns:
- `question`: The mathematical problem to solve
- `answer`: The expected numerical answer
- `correct_answer`: The full solution/explanation (optional)

Place the dataset file at: `hrl-reasoning/dataset/MATH.csv`

## Configuration

The training configuration is specified in `scripts/config/archer_math.yaml`. Key parameters include:

- `env_name`: The environment class to use (`LLMBatchedMathEnv`)
- `policy_lm`: The language model to use for the agent (e.g., "meta-llama/Llama-3.2-3B-Instruct")
- `critic_lm`: The model for critic network (usually "roberta-base")
- `max_tokens`: Maximum number of tokens for environment steps
- `rollout_size`: Number of parallel environments
- `batch_size`: Training batch size
- `iterations`: Number of training iterations
- `temperature`: Temperature for token generation

## Training

To start training ArCHer on the MATH dataset, run:

```bash
python scripts/run.py
```

The training process includes:
1. Environment initialization with the MATH dataset
2. Agent initialization with the specified language model
3. Collection of trajectories through environment interaction
4. Training the critic and actor networks
5. Periodic evaluation and checkpoint saving

## Monitoring

Training progress is logged through:
- Console output showing rewards and loss values
- Weights & Biases (wandb) integration for visualization (if configured)
