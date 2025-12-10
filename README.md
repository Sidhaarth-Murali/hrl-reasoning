## Repo for MLMT-RL — Reasoning with textual feedback

This repo contains multiple reasoning RL trainers. 
**MLMT-RL**: a two-level, two-turn loop where a lower solver tries, a higher feedback writer critiques, and the solver refines. It includes full-grad and stop-grad variants. Baselines (ArCHer, SCoRe, RL-Guided SCoRe) are provided for comparison. Supports math (MATH) and code (MBPP).

### Concept: MLMT-RL in brief
Episode with one reward R on the refined answer ŷ:
1) Turn 1 (solver): sample z ~ π_L(·|x).
2) Turn 2 (feedback): sample g ~ π_H(·|x, z).
3) Turn 3 (refine): sample ŷ ~ π_L(·|x, z, g); verifier returns R.
4) Train:
   - Solver: policy loss ∝ (R − V_L)·(logπ_L(z)+logπ_L(ŷ)) and value loss (V_L − R)².
   - Feedback: policy loss uses R plus optional value shaping; stop-grad variant blocks value backprop into feedback.

### Methods supported
- **MLMT-RL / BiLevel SCoRe**: solver + feedback, with value baseline; full-grad or stop-grad.
- **RL-Guided SCoRe**: two-turn, trains guidance model, no bi-level value shaping.
- **SCoRe**: two-turn self-correction, KL-regularized, no guidance training.
- **ArCHer**: single-turn actor-critic baseline.

### Quick setup
```bash
git clone https://github.com/Sidhaarth-Murali/hrl-reasoning.git
cd hrl-reasoning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Auth if needed:
- HF hub: `huggingface-cli login`
- wandb (optional): `wandb login`
Datasets expected under `dataset/` (e.g., `MATH.csv`, `mbpp_filtered.csv`).

### Entry points
- MLMT-RL: `python run_bi_level_score.py`
- Generic driver (all agents): `python scripts/run.py --config-name <config> --env-type <math|code>`

### How to run (examples)
- MLMT-RL, math, full-grad (default):
  ```bash
  python run_bi_level_score.py hydra.run.dir=outputs/mlmt_math save_path=outputs/mlmt_math/checkpoints
  ```
- MLMT-RL, math, stop-grad guidance:
  ```bash
  python run_bi_level_score.py stop_value_gradients=true
  ```
- SCoRe (no guidance training):
  ```bash
  python scripts/run.py --config-name score_math --env-type math
  ```
- RL-Guided SCoRe (guidance trained):
  ```bash
  python scripts/run.py --config-name score_math --env-type math train_guidance_model=true use_smart_corrections=true
  ```
- ArCHer baseline:
  ```bash
  python scripts/run.py --config-name archer_math --env-type math
  ```
- Code tasks: use `--env-type code` with code configs (e.g., `bi_level_score_code.yaml`, `archer_code.yaml`).

### Configs (Hydra)
Location: `scripts/config/`
- MLMT-RL: `math_configs/bi_level_score_math.yaml`, `code_configs/bi_level_score_code.yaml`
- Baselines: `archer_*`, `score_*`, `rl_guided_score_*`

Key knobs:
- Model/rollout: `policy_lm`, `max_tokens`, `rollout_size`, `batch_size`
- MLMT-RL: `value_coef`, `stop_value_gradients`, `train_guidance_model`, `guidance_lr`, `guidance_kl_coef`
- Logging/checkpoints: `use_wandb`, `save_path`, `save_freq`

### Flow (end to end)
1) Load env (math/code) and agent (solver LM; optional guidance LM).
2) Turn 1: zero-shot attempt (template in `archer/prompts/math.py` or code equivalent).
3) Turn 2: generate feedback (guidance model or template).
4) Turn 3: refine with feedback, get reward from verifier.
5) Train:
   - Solver policy with advantage (value baseline) and value MSE.
   - Guidance policy (if enabled) with optional value shaping; stop-grad blocks value backprop.
6) Log metrics; save to `save_path`.

### Outputs
- Checkpoints in `save_path/` (solver/guidance/value as applicable)
- wandb logs if enabled
- Console prints of config and mode (full-grad vs stop-grad)

### Repo map (key dirs)
- `archer/environment/` : math/code envs and verifiers
- `archer/models/`      : solver LM wrapper, critic, value
- `archer/algorithms/`  : trainers (archer, score, rl_guided, bi_level)
- `scripts/config/`     : Hydra configs
- Entry scripts         : `run_bi_level_score.py`, `scripts/run.py`

### Notes
- Keep secrets (HF, wandb) in env vars; don’t commit tokens.
- Use GPUs; if memory is tight, reduce `rollout_size` or `batch_size`.
