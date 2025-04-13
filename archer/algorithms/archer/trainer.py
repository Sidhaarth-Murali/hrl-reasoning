import torch
import transformers
from tqdm import tqdm
from torch.utils.data import DataLoader
from archer.data import DummyDataset
import copy
import threading
from typing import Tuple
import random
import copy
import time
def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict
class ArcherTrainer():
    def __init__(self, agent,\
                 accelerator,\
                    tokenizer,\
                    critic_lr: float = 1e-3,\
                    lm_lr: float = 1e-5,\
                    grad_accum_steps: int = 8,\
                    gamma: float = 0.9,
                    tau: float = 0.1,
                    epochs: int = 3,
                    max_grad_norm: float=0.01,
                    actor_epochs: int = 3):
        """
        beta: coefficient for the bc loss
        """
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr = lm_lr)
        self.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr = critic_lr)
        self.criterion = torch.nn.MSELoss()
        self.grad_accum_steps = grad_accum_steps
        self.actor_epochs = actor_epochs
        self.gamma = gamma
        self.epochs = epochs
        self.step = 0
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.critic_optimizer, self.lm_optimizer = self.accelerator.prepare(self.critic_optimizer, self.lm_optimizer)

    def critic_loss(self, observation, action, reward, next_observation, done, mc_return,**kwargs):
        reward = torch.Tensor(reward).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype).flatten()
        done = torch.Tensor(done).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype).flatten()
        q1, q2, v1, v2 = self.agent.critic(observation, action, detach_model=False)
        # print("finish one forward pass")
        with torch.no_grad():
            pi_action = self.agent.get_action(copy.deepcopy(observation))
            # target_q1, target_q2 = self.agent.get_q(observation, pi_action, detach_model=False)
            target_q1, target_q2, _ , _ = self.agent.target_critic(copy.deepcopy(observation), pi_action, detach_model=False)
        q1 = q1.flatten()
        q2 = q2.flatten()
        v1 = v1.flatten()
        v2 = v2.flatten()
        target_q1 = target_q1.flatten()
        target_q2 = target_q2.flatten()
        with torch.no_grad():
            #action is dummy here
            _, _ , target_v1, target_v2 = self.agent.target_critic(next_observation, copy.deepcopy(action))
            target_v1 = reward + (1 - done)*target_v1.flatten()*self.gamma
            target_v2 = reward + (1 - done)*target_v2.flatten()*self.gamma
        # target_v1 = torch.zeros_like(q1)
        # target_v2 = torch.zeros_like(q2)
        q1_loss = self.criterion(q1, target_v1)
        q2_loss = self.criterion(q2, target_v2)
        v1_loss = self.criterion(v1, target_q1)
        v2_loss = self.criterion(v2, target_q2)
        self.accelerator.backward((q1_loss+q2_loss+v1_loss+ v2_loss))
        q1_loss, q2_loss, v1_loss, v2_loss = q1_loss.detach().cpu(), q2_loss.detach().cpu(),\
                                             v1_loss.detach().cpu(), v2_loss.detach().cpu()
        q1, q2, v1, v2, target_q1, target_q2 = q1.detach().cpu(), q2.detach().cpu(), v1.detach().cpu(),\
                                            v2.detach().cpu(), target_q1.detach().cpu(), target_q2.detach().cpu()
        return {"q1.loss": q1_loss,\
                    "q2.loss": q2_loss,\
                    "v1.loss": v1_loss,\
                    "v2.loss": v2_loss,\
                    "q1.mean": torch.mean(q1),\
                    "q1.min": torch.min(q1),\
                    "q1.max": torch.max(q1),\
                    "q1.std": torch.std(q1),\
                    "q2.mean": torch.mean(q2),
                    "q2.max": torch.max(q2),
                    "q2.min": torch.min(q2),
                    "q2.std": torch.std(q2),\
                    "v1.mean": torch.mean(v1),\
                    "v1.min": torch.min(v1),\
                    "v1.max": torch.max(v1),\
                    "v1.std": torch.std(v1),
                    "v2.mean": torch.mean(v2),
                    "v2.max": torch.max(v2),
                    "v2.min": torch.min(v2),
                    "v2.std": torch.std(v2),
                    "target_q1.mean": torch.mean(target_q1),\
                    "target_q1.min": torch.min(target_q1),\
                    "target_q1.max": torch.max(target_q1),\
                    "target_q1.std": torch.std(target_q1),
                    "target_q2.mean": torch.mean(target_q2),
                    "target_q2.max": torch.max(target_q2),
                    "target_q2.min": torch.min(target_q2),
                    "target_q2.std": torch.std(target_q2),}

    def actor_loss(self, observation, pi_action, advantage, **kwargs):
        # with torch.no_grad():
        #     pi_action = self.agent.get_action(observation)
        # breakpoint()
        # action = [a if random.random()>0.5 else pi_a for a, pi_a in zip(batch_action, pi_action)]
        action = pi_action
        log_prob = self.agent.get_log_prob(observation, action)
        advantage = torch.Tensor(advantage).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype)
        #in the case where a baseline is used
        if isinstance(log_prob, Tuple):
            values, log_prob, mask = log_prob
            values = values.squeeze(-1)
            advantage = advantage.reshape(-1, 1).broadcast_to(values.size())
            value_loss = torch.mean(((advantage - values)*mask)**2)
            with torch.no_grad():
                residual_advantage = advantage - values
            pg_loss = -torch.mean(torch.sum(residual_advantage*log_prob*mask, dim = 1))

        else:
            advantages = advantage.flatten()
            values = torch.zeros_like(advantages)
            residual_advantage = torch.zeros_like(advantages)
            # ratio = torch.exp(log_prob - old_log_prob).flatten()
            # pg_loss1 = -advantages*ratio
            # pg_loss2 = -advantages*torch.clip(ratio, 1 - self.clip_range, 1+ self.clip_range)
            # pg_loss = torch.mean(torch.maximum(pg_loss1, pg_loss2))
            pg_loss = -torch.mean(log_prob.flatten()*advantages)
            value_loss = torch.zeros_like(pg_loss)
        advantages = advantage.flatten()
        self.accelerator.backward(pg_loss+value_loss)
        advantages = advantages.detach().cpu()
        return {"pg.loss": pg_loss.detach().cpu().item(),
                "values.loss": value_loss.detach().cpu().item(),
                "values.mean": values.mean(),
                "values.max": torch.max(values),
                "values.min": torch.min(values),
                "values.std": torch.std(values),
                "advantages.mean": advantages.mean(),
                "advantages.max": torch.max(advantages),
                "advantages.min": torch.min(advantages),
                "advantages.std": torch.std(advantages),
                "residual_advantages.mean": residual_advantage.mean(),
                "residual_advantages.max": torch.max(residual_advantage),
                "residual_advantages.min": torch.min(residual_advantage),
                "residual_advantages.std": torch.std(residual_advantage),}

    def update(self, replay_buffer, no_update_actor=False):
        self.step += 1
        info = {}
        critic_info_list = []
        
        # --- Critic Update ---
        # Pre-sample data once for the epoch to reduce overhead.
        data = [replay_buffer.sample(1) for _ in range(self.grad_accum_steps * replay_buffer.batch_size)]
        for d in data:
            for k, v in d.items():
                d[k] = v[0]
        # Increase num_workers for faster data loading.
        dataloader = DataLoader(DummyDataset(data), batch_size=replay_buffer.batch_size, num_workers=4, shuffle=True)
        dataloader = self.accelerator.prepare(dataloader)
        
        
        self.critic_optimizer.zero_grad()
        for batch in tqdm(dataloader, desc="Critic Update", leave=False):
            critic_info_list.append(self.critic_loss(**batch))
        self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        # Soft-update target critic after critic updates.
        self.agent.soft_update_target_critic(tau=self.tau)
        
        info.update(dict_mean(critic_info_list))
        
        # --- Actor Update ---
        if not no_update_actor:
            print(">>> Updating actor")
            actor_info_list = []
            # Pre-sample data once for the actor update.
            data = [replay_buffer.sample(1) for _ in range(self.grad_accum_steps * replay_buffer.batch_size)]
            for d in data:
                for k, v in d.items():
                    d[k] = v[0]
            dataloader = DataLoader(DummyDataset(data), batch_size=replay_buffer.batch_size, num_workers=4, shuffle=False)
            dataloader = self.accelerator.prepare(dataloader)
            
            self.lm_optimizer.zero_grad()
            for batch in tqdm(dataloader, desc="Actor Update", leave=False):
                # Calculate advantages in a no_grad block to save memory.
                with torch.no_grad():
                    pi_action = self.agent.get_action(batch["observation"])
                    q1, q2, v1, v2 = self.agent.critic(batch["observation"], pi_action)
                    q = torch.minimum(q1, q2)
                    v = torch.minimum(v1, v2)
                    advantages = q - v
                actor_info_list.append(self.actor_loss(**batch, pi_action=pi_action, advantage=advantages))
            self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.lm_optimizer.step()
            
            info.update(dict_mean(actor_info_list))
            
        return info


    def save(self, path):
        torch.save({'model_state_dict': self.accelerator.unwrap_model(self.agent.model).state_dict(),
                    'critic_state_dict': self.accelerator.unwrap_model(self.agent.critic).state_dict(),
                    'target_critic_state_dict': self.accelerator.unwrap_model(self.agent.target_critic).state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'lm_optimizer_state_dict': self.lm_optimizer.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.agent.model.load_state_dict(checkpoint['model_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.agent.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.lm_optimizer.load_state_dict(checkpoint['lm_optimizer_state_dict'])
        return self.agent