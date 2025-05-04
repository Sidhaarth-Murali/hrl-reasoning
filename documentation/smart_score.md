# SMART-SCoRe

This document explains the SMART-SCoRe implementation, which enhances the standard SCoRe (Self-Correction via Reinforcement Learning) approach by using dynamic, context-aware guidance for the correction phase.

## Overview

In standard SCoRe, the correction phase uses a static template to prompt the model to reconsider its solution. SMART-SCoRe improves this by generating tailored correction guidance based on analyzing the specific errors in the initial solution.

## How It Works

1. The base SCoRe model still learns through the same two-stage process:
   - Stage I: Learn to fix mistakes with KL penalty on the first turn
   - Stage II: Joint multi-turn RL with reward shaping

2. The key innovation is in the correction prompt generation:
   - Input: Mathematical problem and initial solution
   - Process: A higher-level model analyzes the solution for errors
   - Output: Custom correction guidance that targets specific mistakes
   - The guidance is more effective because it addresses the particular misconceptions present

3. The training loop follows the same pattern:
   - Collect first-turn trajectories
   - Generate smart correction prompts
   - Collect second-turn trajectories with these prompts
   - Update the model using the SCoRe algorithm

## Configuration

Use the `smart_score_math.yaml` config file to train with SMART-SCoRe:

```yaml
defaults:
  - score_math
  - _self_

# SMART_SCoRe specific settings
checkpoint_path: null
use_smart_corrections: true
correction_model_path: null   

# use a meaningful run name for tracking
run_name: 'smart-score-math'
project_name: 'score_math'
```

## Training

To train a model with SMART-SCoRe, run:

```bash
python -m archer.main config/smart_score_math.yaml
```

The training process will:
1. Use the specified correction model (or the base model if none provided)
2. Generate custom correction guidance for each problem-solution pair
3. Train the base model using the standard SCoRe approach

## Implementation Details

Key implementation details include:

1. `generate_smart_correction_prompt`: Analyzes the initial solution and generates specific guidance
2. Batched processing for memory efficiency
3. Graceful fallback to standard templates if dynamic generation fails
4. Support for different correction models (can use the same model for both tasks if memory is limited)

## Benefits

Using SMART-SCoRe provides several advantages:

1. Higher quality correction - guidance targets specific issues rather than generic prompting
2. Better learning signal - the model learns from more precise feedback
3. Improved performance - models trained with SMART-SCoRe can achieve better results
4. Mimics real-world tutoring - similar to how a teacher would provide targeted feedback 