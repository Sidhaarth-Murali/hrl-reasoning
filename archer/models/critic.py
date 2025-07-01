import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import Tuple
import torch.nn as nn
import numpy as np
from transformers import RobertaTokenizer, RobertaModel

class DoubleCritic(torch.nn.Module):
    def __init__(self, device, accelerator, critic_lm, cache_dir, in_dim, out_dim):
        super(DoubleCritic, self).__init__()
        self.device = device
        self.accelerator = accelerator
        self.base_lm = AutoModel.from_pretrained(critic_lm, cache_dir=cache_dir).to(device)
        self.base_tokenizer = AutoTokenizer.from_pretrained(critic_lm, cache_dir=cache_dir)
        self.base_tokenizer.truncation_side = 'right'  # Keep beginning of math problems
        
        # Math-specific attention layer
        self.math_attention = nn.MultiheadAttention(in_dim, num_heads=8).to(device)
        
        # Enhanced critic networks
        self.critic1 = nn.Sequential(
            nn.Linear(in_dim*2, in_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim, in_dim//2),
            nn.LeakyReLU(),
            nn.Linear(in_dim//2, out_dim)
        ).to(device)
        
        self.critic2 = nn.Sequential(
            nn.Linear(in_dim*2, in_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim, in_dim//2),
            nn.LeakyReLU(),
            nn.Linear(in_dim//2, out_dim)
        ).to(device)
        
        # Enhanced value networks
        self.v_critic1 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim, in_dim//2),
            nn.LeakyReLU(),
            nn.Linear(in_dim//2, out_dim)
        ).to(device)
        
        self.v_critic2 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim, in_dim//2),
            nn.LeakyReLU(),
            nn.Linear(in_dim//2, out_dim)
        ).to(device)
    
    # def prepare(self):
    #     self.base_lm, self.critic1, self.critic2, self.v_critic1, self.v_critic2 = \
    #         self.accelerator.prepare(self.base_lm, self.critic1, self.critic2, self.v_critic1, self.v_critic2)

    def forward(self, observation, action, detach_model=False):
        state_actions = [o + a for o,a in zip(observation, action)]
        
        # Memory optimization: handle device placement efficiently
        obs_ids = self.base_tokenizer(observation, padding=True, return_tensors='pt', max_length=512, truncation=True)
        
        # Only move to device if not already there
        if obs_ids['input_ids'].device != self.device:
            obs_ids = {k: v.to(self.device) for k, v in obs_ids.items()}
        
        # breakpoint() # For debugging tokenization
        if detach_model:
            with torch.no_grad():
                lm_states = self.base_lm(**obs_ids).pooler_output
        else:
            lm_states = self.base_lm(**obs_ids).pooler_output
            
        action_ids = self.base_tokenizer(action, padding=True, return_tensors='pt', max_length=512, truncation=True)
        
        # Only move to device if not already there
        if action_ids['input_ids'].device != self.device:
            action_ids = {k: v.to(self.device) for k, v in action_ids.items()}
        
        # breakpoint() # For debugging tokenization
        if detach_model:
            with torch.no_grad():
                action_states = self.base_lm(**action_ids).pooler_output
        else:
            action_states = self.base_lm(**action_ids).pooler_output
            
        # Memory optimization: clear tokenizer outputs immediately
        del obs_ids, action_ids
        
        # Compute Q-values and V-values before clearing states
        q_states = torch.cat([lm_states, action_states], dim=1)
        q1_output = self.critic1(q_states)
        q2_output = self.critic2(q_states)
        v1_output = self.v_critic1(lm_states)
        v2_output = self.v_critic2(lm_states)
        
        # Clear intermediate states to save memory
        del lm_states, action_states, q_states
        
        return q1_output, q2_output, v1_output, v2_output
    
    def get_q(self, observation, action, detach_model=False):
        """
        Get Q-values from both critics for observation-action pair.
        The actual min operation happens in trainer.py.
        """
        q1, q2, _, _ = self.forward(observation, action, detach_model=detach_model)
        return q1, q2

    def get_v(self, observation, detach_model=False):
        """
        Get V-values from both value functions for observation.
        The actual min operation happens in trainer.py.
        """
        # Process observation through the base model
        obs_ids = self.base_tokenizer(observation, padding=True, return_tensors='pt', 
                                   max_length=512, truncation=True).to(self.device)
        
        # breakpoint() # For debugging observation processing
        if detach_model:
            with torch.no_grad():
                lm_states = self.base_lm(**obs_ids).pooler_output
        else:
            lm_states = self.base_lm(**obs_ids).pooler_output
            
        # Get value estimates from both critics
        v1 = self.v_critic1(lm_states)
        v2 = self.v_critic2(lm_states)
        
        return v1, v2