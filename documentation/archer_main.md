## Theoretical Overview of ArCHer: From Mathematics to Implementation

**ArCHer (Actor-Critic with Hierarchical structures)** brings reinforcement learning (RL) principles into the realm of language model agents. This document outlines the mathematical foundations and their implementation correspondence.

---

### 1. Foundational Reinforcement Learning

The goal in RL is to learn a policy $\pi$ that maximizes expected cumulative discounted rewards:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^T \gamma^t r_t \right]$$

- $\tau$: trajectory  
- $\gamma$: discount factor  
- $r_t$: reward at time $t$

---

### 2. Actor-Critic Framework

ArCHer uses a two-part architecture:

- **Actor (Policy)**: $\pi_\theta(a|s)$, a language model  
- **Critic (Value Estimator)**: with parameters $\phi$

It estimates:

- State Value: $V_\phi(s)$  
- Action Value: $Q_\phi(s, a)$

**Double Critic Setup**:
To reduce overestimation:

- $Q_{\phi_1}(s, a)$, $Q_{\phi_2}(s, a)$  
- $V_{\phi_1}(s)$, $V_{\phi_2}(s)$

---

### 3. Training Objectives

#### Critic Loss:

$$L_{\text{critic}}(\phi) = L_{Q_1} + L_{Q_2} + L_{V_1} + L_{V_2}$$

- Q-value losses:
$$L_{Q_i} = \mathbb{E}_{(s, a, r, s') \sim D} \left[ \left( Q_{\phi_i}(s, a) - \left( r + \gamma V_{\phi_i'}(s') \right) \right)^2 \right]$$

- V-value losses:
$$L_{V_i} = \mathbb{E}_{s \sim D} \left[ \left( V_{\phi_i}(s) - Q_{\phi_i}(s, a') \right)^2 \right]$$
Where $a' \sim \pi_\theta(\cdot|s)$

**Code Reference:**
```python
q1_loss = self.criterion(q1, target_v1)
q2_loss = self.criterion(q2, target_v2)
v1_loss = self.criterion(v1, target_q1)
v2_loss = self.criterion(v2, target_q2)
```

#### Actor Loss:

$$L_{\text{actor}}(\theta) = - \mathbb{E}_{s, a \sim \pi_\theta} \left[ \log \pi_\theta(a|s) \cdot A(s, a) \right]$$

Where advantage is:

$$A(s,a) = \min(Q_{\phi_1}(s,a), Q_{\phi_2}(s,a)) - \min(V_{\phi_1}(s), V_{\phi_2}(s))$$

**Code Reference:**
```python
q = torch.minimum(q1, q2)
v = torch.minimum(v1, v2)
advantages = q - v
pg_loss = -torch.mean(log_prob.flatten() * advantages)
```

---

### 4. Hierarchical Optimization

ArCHer supports:
- **Utterance-level**: Complete text as an action
- **Token-level**: Fine-grained optimization using token-level log probs and masks

```python
if isinstance(log_prob, Tuple):
    values, log_prob, mask = log_prob
    pg_loss = -torch.mean(torch.sum(residual_advantage * log_prob * mask, dim=1))
```

---

### 5. Practical Implementation Highlights

#### Target Network Update:
Soft update using Polyak averaging:

$$\phi'_{\text{target}} \leftarrow \tau \phi + (1 - \tau) \phi'_{\text{target}}$$

```python
for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
```

#### Statistics Tracked:
- Mean, min, max, std for: $Q$, $V$, $A$
- Loss values: $L_{Q_i}$, $L_{V_i}$, $L_{\text{actor}}$

Used for:
- Stability tracking
- Over/underestimation checks
- Hyperparameter tuning

---

### 6. Dry Run Example: Twenty Questions

**Initial State**: "Questions:\n", Target: "cat"

| Step | Description |
|------|-------------|
| 1 | Actor generates: "Is it an animal?" |
| 2 | Env responds: "Yes.", $r = -1$, done = False |
| 3 | Critic evaluates:<br> $Q_1 = 0.25$, $Q_2 = 0.30$<br> $V_1 = 0.15$, $V_2 = 0.20$<br> Next: "Does it have fur?"<br> Target Qs: $0.45$, $0.50$<br> $\text{Target V}_1 = -0.685$, $\text{Target V}_2 = -0.640$<br> Losses:<br> $Q_1$ loss = $0.874$, $Q_2$ loss = $0.884$<br> $V_1$ loss = $0.090$, $V_2$ loss = $0.090$<br> Total: **$1.938$** |
| 4 | Actor log prob: $-2.3$<br> Advantage: $0.10$<br> PG loss: $0.23$ |
| 5 | Target Update with $\tau = 0.1$ |

---

### 7. Mathematical ↔ Code Mapping

| Concept | Code |
|--------|------|
| State $s$ | `obs_ids = tokenizer(observation)` |
| Action $a \sim \pi_\theta$ | `model.generate(**obs_ids)` |
| Log Prob $\log \pi_\theta(a|s)$ | `log_prob = torch.sum(log(...))` |
| Advantage $A(s,a)$ | `advantages = q - v` |
| Backpropagation | `accelerator.backward(...)` |
| Stats | `return {"q1.mean": ..., "v1.std": ...}` |

---

### 8. Key Theoretical Innovations

- **Hierarchical Representation**: Combines token + utterance levels to handle sparse rewards.
- **Double Critics**: Prevents Q-value overestimation.
- **Soft Actor-Critic (SAC) Adaptation**: Transfers continuous control ideas to discrete language spaces.
- **Target Networks**: Smooth training dynamics.

These innovations allow ArCHer to train language model agents capable of multi-turn interactive behavior by optimizing long-term reward signals.

