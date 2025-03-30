# Behavior Cloning in Archer Framework

This document provides a detailed explanation of Behavior Cloning (BC) as implemented in the Archer framework, including both the standard implementation and the filtered BC variant.

## Table of Contents
1. [Introduction to Behavior Cloning](#introduction-to-behavior-cloning)
2. [Mathematical Formulation](#mathematical-formulation)
3. [Implementation Details](#implementation-details)
4. [Code Walkthrough](#code-walkthrough)
5. [Running Instructions](#running-instructions)
6. [Performance and Evaluation](#performance-and-evaluation)

## Introduction to Behavior Cloning

Behavior Cloning (BC) is a straightforward supervised learning approach to imitation learning where an agent learns a policy by directly mimicking expert demonstrations.

In the context of the Archer framework and mathematical reasoning:
- We use BC to train a language model to generate appropriate solution steps for mathematical problems
- The training occurs by showing the model examples of problems and their solutions
- The model learns to map observations (math problems) to actions (solution steps)

Unlike reinforcement learning which optimizes based on rewards, BC directly trains a policy to predict the actions an expert would take in the same state.

### Advantages of BC
- Conceptually simple and easy to implement
- Directly leverages existing demonstrations
- Stable training process
- No need for complex reward engineering

### Limitations of BC
- Can suffer from compounding errors and distribution shift
- Performance limited by demonstration quality
- May not generalize well to unseen situations

## Mathematical Formulation

Behavior cloning frames the policy learning problem as a supervised learning task. Given a dataset of observation-action pairs $\mathcal{D} = \{(o_i, a_i)\}_{i=1}^N$ where $o_i$ is an observation and $a_i$ is the corresponding action, BC aims to learn a policy $\pi_\theta(a|o)$ parameterized by $\theta$ that maximizes the likelihood of the actions in the dataset.

### The BC Objective
The standard BC objective is to minimize the negative log-likelihood:

$$\mathcal{L}_{BC}(\theta) = -\mathbb{E}_{(o,a) \sim \mathcal{D}} [\log \pi_\theta(a|o)]$$

For language models, this translates to predicting the tokens of the action given the observation. Given a tokenized observation $x = (x_1, x_2, ..., x_n)$ and action $y = (y_1, y_2, ..., y_m)$, the loss becomes:

$$\mathcal{L}_{BC}(\theta) = -\sum_{i=1}^{m} \log P_\theta(y_i | x, y_{<i})$$

### Filtered BC

In our framework, we also implement a variant called Filtered BC which:
1. Collects trajectories through environment interaction
2. Filters trajectories based on rewards (keeping only the top 10% performing ones)
3. Trains the policy using BC on these filtered examples

This adds an additional filtering step:

$$\mathcal{D}_{\text{filtered}} = \{(o, a) \in \mathcal{D} | R((o, a)) \geq \text{cutoff}\}$$

where $R((o, a))$ is the reward for the trajectory containing the (observation, action) pair, and cutoff is typically the 90th percentile of rewards.

## Implementation Details

The Archer framework implements BC in two ways:

1. **Simple BC Implementation**: Direct supervised learning, implemented in `archer/algorithms/bc/`
   - Uses `plain_bc_loss` for token prediction
   - Simple training loop without reinforcement learning components
   - Suitable for quick training on static datasets

2. **Filtered BC Implementation**: Integrated with RL, in `archer/algorithms/online_filteredbc/`
   - Uses the same underlying BC loss
   - Extends with filtering based on reward
   - Integrates with the main RL training loop
   - Supports online data collection

### Core Components

1. **BCTrainer**: Main class handling the BC training process
2. **plain_bc_loss**: Loss function implementing the BC objective
3. **train_loop**: Simple training loop for pure BC
4. **offpolicy_train_loop**: Complex loop for online Filtered BC

## Code Walkthrough

### BC Loss Function

The core BC loss is implemented in `archer/algorithms/bc/core.py`:

```python
def plain_bc_loss(model, tokenizer, observation, action, **kwargs):
    """Calculate the behavior cloning loss."""
    # Tokenize observation and action
    action_ids = tokenizer(action, return_tensors='pt', padding=True).to(model.device)
    obs_ids = tokenizer(observation, return_tensors='pt', padding=True).to(model.device)
    
    # Concatenate input_ids and attention_mask
    input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim=1)
    attention_mask = torch.cat([obs_ids["attention_mask"], action_ids["attention_mask"]], dim=1)
    
    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    prediction_probs = softmax(outputs.logits)
    
    # Extract probabilities for the chosen tokens
    selected_prediction_probs = torch.take_along_dim(
        prediction_probs[:, obs_ids["attention_mask"].size(1)-1:-1], 
        action_ids["input_ids"].unsqueeze(2), 
        dim=2
    ).squeeze(2)
    
    # Calculate log probabilities and sum
    logsum_probs = torch.sum(torch.log(selected_prediction_probs)*action_ids["attention_mask"], dim=1)
    
    # Return negative mean log probability (loss)
    return -logsum_probs.mean()
```

This function:
1. Tokenizes the observation and action
2. Concatenates them to form the input sequence
3. Runs a forward pass through the model
4. Calculates the token prediction probabilities
5. Computes the negative log likelihood loss

### BCTrainer Class

The `BCTrainer` class (in `archer/algorithms/online_filteredbc/trainer.py`) handles the behavior cloning training process:

```python
class BCTrainer():
    def __init__(self, agent, tokenizer, accelerator, 
                 lm_lr=1e-5, epochs=3, max_grad_norm=0.01, grad_accum_steps=8):
        """Initialize the BC trainer."""
        self.agent = agent
        self.tokenizer = tokenizer
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr=lm_lr)
        self.critic_optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
        self.criterion = torch.nn.MSELoss()
        self.grad_accum_steps = grad_accum_steps
        self.epochs = epochs
        self.step = 0
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.agent, self.lm_optimizer, self.critic_optimizer = self.accelerator.prepare(
            self.agent, self.lm_optimizer, self.critic_optimizer
        )

    def actor_loss(self, observation, action, **kwargs):
        """Calculate the BC loss for the actor."""
        loss = plain_bc_loss(
            self.accelerator.unwrap_model(self.agent).model, 
            self.tokenizer, 
            observation, 
            action
        )
        self.accelerator.backward(loss)
        return {"bc.loss": loss.detach().cpu().item()}

    def update(self, replay_buffer, no_update_actor=False):
        """Update the policy using BC loss."""
        self.step += 1
        info = {}
        info_list = []
        
        if not no_update_actor:
            action_bsize = 1 if 'llama' in self.accelerator.unwrap_model(self.agent).policy_lm else replay_buffer.batch_size
            
            for _ in range(self.epochs):
                self.lm_optimizer.zero_grad()
                data = [replay_buffer.sample(1) for _ in range(self.grad_accum_steps*replay_buffer.batch_size)]
                
                for d in data:
                    for k,v in d.items():
                        d[k] = v[0]
                
                dataloader = DataLoader(DummyDataset(data), batch_size=action_bsize, shuffle=False)
                dataloader = self.accelerator.prepare(dataloader)
                
                for batch in dataloader:
                    info_list.append(self.actor_loss(**batch))
                
                self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.lm_optimizer.step()
        
        info.update(dict_mean(info_list))
        return info
```

This class:
1. Configures the optimizer and training parameters
2. Implements the actor_loss method for computing BC loss
3. Provides an update method for training on batches from a replay buffer

### Filtered BC Training Loop

In Filtered BC, the training process happens in `offpolicy_train_loop` in `archer/algorithms/offpolicy_train_loop.py`:

```python
# Simplified version of the filtered BC section
if 'filtered' in agent_type.lower():
    filtered_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
    episode_rewards = [d[0]["trajectory_reward"] for d in all_trajectories]
    cutoff = np.quantile(episode_rewards, 1 - 0.1)  # 90th percentile
    print("Episode Reward Cutoff: ", cutoff)
    
    # Filter trajectories by reward
    filtered_trajectories = list(filter(
        lambda x: x[0]["trajectory_reward"] >= cutoff, 
        all_trajectories
    ))
    
    # Extract individual transitions
    data = sum(filtered_trajectories, [])
    
    # Add to buffer
    for d in data:
        filtered_buffer.insert(**d)
    
    # Update the policy
    info.update(trainer.update(filtered_buffer, no_update_actor=(i < warmup_iter)))
```

This section:
1. Calculates a reward cutoff (90th percentile)
2. Filters trajectories that exceed this cutoff
3. Populates a filtered replay buffer with the transitions
4. Updates the policy using behavior cloning on this filtered data

## Running Instructions

There are two ways to run BC in the Archer framework:

### 1. Running Simple BC

This uses the direct BC implementation without reinforcement learning components:

```bash
# Run the script directly
python scripts/run_bc.py --config bc_math

# Or use the convenience script
./scripts/run_simple_bc.sh
```

This will:
- Load math problems from the dataset
- Generate solutions using the environment's model
- Train a policy model using behavior cloning
- Save the model to the configured directory

### 2. Running Filtered BC

This uses BC within the RL framework, with reward-based filtering:

```bash
# Run with the online_filteredbc agent type
python scripts/run.py --config bc_math

# Or use the convenience script
./scripts/run_bc.sh
```

This will:
- Initialize the environment and agent
- Collect trajectories through environment interaction
- Filter the trajectories based on rewards
- Train the policy using BC on the filtered examples
- Periodically evaluate and save checkpoints

### Configuration

The BC configuration is specified in `scripts/config/bc_math.yaml`. Key parameters include:

```yaml
# Agent type - use online_filteredbc for filtered BC
agent_type: "online_filteredbc"

# Model settings
policy_lm: "meta-llama/Llama-3.2-3B-Instruct"
critic_lm: "roberta-base"

# Training hyperparameters
rollout_size: 16      # Number of parallel environments
batch_size: 2         # Training batch size
iterations: 250       # Total iterations
actor_epochs: 3       # BC updates per batch
grad_accum_steps: 1   # Gradient accumulation steps
lm_lr: 1e-6           # Learning rate for policy
```

## Performance and Evaluation

To evaluate BC performance:

1. **During Training**: Monitor the BC loss curves
   - Decreasing loss indicates the model is learning to mimic the demonstrations
   - Wandb logs can be viewed for detailed metrics

2. **After Training**: Test on held-out problems
   - The model should generate reasonable solution steps
   - Compare with the original solutions for accuracy

3. **Comparison with RL**: BC typically:
   - Converges faster than pure RL methods
   - May plateau at lower performance than well-tuned RL
   - Shows more stable learning curves

### Example Output

After training, you can test the model by loading the saved checkpoint and running inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_path = "/home/pramit/hrl-nips-work/hrl-reasoning/.saved_models/bc_math/bc_model.pt"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
checkpoint = torch.load(model_path)

# Initialize model and load weights
model = AutoModelForCausalLM.from_pretrained(model_name)
model.load_state_dict(checkpoint['model_state_dict'])
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example problem
problem = "Find all values of x such that 2x^2 - 5x + 2 = 0."

# Generate solution
inputs = tokenizer(problem, return_tensors="pt")
outputs = model.generate(
    inputs.input_ids, 
    max_new_tokens=100,
    temperature=0.7
)
solution = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(solution)
``` 