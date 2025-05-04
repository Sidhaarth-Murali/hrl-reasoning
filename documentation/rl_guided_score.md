# RL-Guided SCoRe

This document explains the RL-Guided SCoRe implementation, which extends the standard SCoRe (Self-Correction via Reinforcement Learning) approach by training not just the base model, but also the guidance model that provides correction hints.

## Overview

In standard SMART-SCoRe, the correction guidance comes from a static model. In RL-Guided SCoRe, we train this guidance model using REINFORCE to provide better correction hints based on whether they help the base model reach the correct answer.

## How It Works

1. The base SCoRe model learns through a two-stage process:
   - Stage I: Learn to fix mistakes with KL penalty on the first turn
   - Stage II: Joint multi-turn RL with reward shaping

2. The guidance model is trained using REINFORCE:
   - Input: Mathematical problem and first-turn solution
   - Output: Correction guidance text
   - Reward: Whether the base model gets the correct answer after following the guidance
   - Loss: Standard policy gradient loss with a small KL penalty

3. The training loop alternates between:
   - Collecting trajectories using the current models
   - Updating both the base model and the guidance model

## Configuration

Use the `rl_guided_score_math.yaml` config file to train with RL-guided SCoRe:

```yaml
defaults:
  - smart_score_math
  - _self_

# Add RL training for guidance model
train_guidance_model: true 
guidance_lr: 1e-6          
guidance_model_path: null    
guidance_kl_coef: 0.05      

run_name: 'rl-guided-score-math'
```

## Training

To train a model with RL-guided SCoRe, run:

```bash
python -m archer.main config/rl_guided_score_math.yaml
```

The training process will:
1. Initialize the guidance model (or use a pre-trained one)
2. Train the base SCoRe model using the standard two-stage approach
3. Train the guidance model to provide better correction hints

## Implementation Details

Key implementation details include:

1. `RLGuidedSCoReTrainer`: Extends `SCoReTrainer` to include guidance model training
2. `generate_custom_guidance`: Generates correction text and tracks log probabilities for RL
3. `train_guidance_model`: Implements REINFORCE for the guidance model
4. Optimized memory usage for training the two models together

## Results

Preliminary results suggest that training the guidance model can lead to:
1. Better quality correction hints that target specific mistakes
2. Improved overall performance on mathematical problem-solving
3. More diverse and tailored guidance compared to static approaches 