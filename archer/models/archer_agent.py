import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import torch.nn as nn
import numpy as np
from archer.models.critic import DoubleCritic


class ArcherAgent(torch.nn.Module):
    def __init__(self, device, accelerator, policy_lm = "gpt2", critic_lm = "roberta-base", 
                cache_dir = '~/.cache', dropout = 0.5, TEMPLATE = None, use_lora=False,
                do_sample = True, temperature = 1.0, max_new_tokens = 32, use_bfloat16 = False, eos_str = '\n'):
        super(ArcherAgent, self).__init__()
        if use_bfloat16:
            self.model = AutoModelForCausalLM.from_pretrained(policy_lm, cache_dir=cache_dir,
                                                              torch_dtype = torch.bfloat16).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(policy_lm, cache_dir=cache_dir).to(device)
        if use_lora:
            from peft import LoraConfig, TaskType, get_peft_model
            lora_config = LoraConfig(
                r=16,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=32,
                lora_dropout=0.05
            )
            self.model = get_peft_model(self.model, lora_config)
            print("Using LoRA")
            self.model.print_trainable_parameters()
        self.template = TEMPLATE
        self.policy_lm = policy_lm
        self.critic = DoubleCritic(device, accelerator, critic_lm = critic_lm, cache_dir = cache_dir, in_dim = 768, out_dim = 1)  
        self.target_critic = DoubleCritic(device, accelerator, critic_lm = critic_lm, cache_dir = cache_dir, in_dim = 768, out_dim = 1) 
        self.soft_update_target_critic(1)
        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True, cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'left'
        if self.tokenizer.pad_token is None:
            special_tokens_dict = {'pad_token': '[PAD]'}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax(dim= -1)
        self.do_sample = do_sample
        self.temperature = temperature
        self.accelerator = accelerator
        self.max_new_tokens = max_new_tokens
        self.eos_str = eos_str
    
    def prepare(self):
        # self.model = self.accelerator.prepare(self.model)
        # self.critic.prepare()
        # self.target_critic.prepare()
        self.model, self.critic, self.target_critic = self.accelerator.prepare(self.model, self.critic, self.target_critic)

    def get_action(self, observation):
        if self.template is not None:
            observation = [self.template.replace("{obs}", obs) for obs in observation]
        
        # Create actions list to return
        actions = []
        
        # Process each observation individually with chat template
        for obs in observation:
            # Define messages for chat template
            messages = [
                {
                    "role": "system",
                    "content": "You are a highly intelligent math assistant. Provide concise, step-by-step solutions to math problems."
                },
                {
                    "role": "user",
                    "content": obs
                },
            ]
            
            # Format input using chat template if available
            if hasattr(self.tokenizer, "apply_chat_template"):
                inputs = self.tokenizer.apply_chat_template(
                    conversation=messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(self.device)
            else:
                # Fallback for older tokenizers: manually format the conversation
                formatted_prompt = f"System: You are a highly intelligent math assistant. Provide concise, step-by-step solutions to math problems.\n\nUser: {obs}\n\nAssistant:"
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            # Generate 4 tokens but only use the first one
            with torch.no_grad():
                outputs = self.accelerator.unwrap_model(self.model).generate(
                    input_ids=inputs["input_ids"] if isinstance(inputs, dict) else inputs,
                    attention_mask=inputs.get("attention_mask") if isinstance(inputs, dict) else None,
                    max_new_tokens=4,  
                    do_sample=self.do_sample,
                    temperature=self.temperature,  
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Get the context length to extract just the first new token
            if isinstance(inputs, dict):
                context_len = inputs["input_ids"].size(1)
            else:
                context_len = inputs.size(1)
            if outputs.size(1) > context_len:
                token_id = outputs[0, context_len]  # Only take the first new token
                token_text = self.tokenizer.decode(token_id.unsqueeze(0), skip_special_tokens=False)
            else:
                token_text = " "
            
            actions.append(token_text)
        
        return actions

    def get_q(self, observation, action, detach_model=False):
        return self.critic.get_q(observation, action, detach_model = detach_model)

    def get_v(self, inputs, detach_model=False):
        return self.critic.get_v(inputs, detach_model = detach_model)
    
    def get_target_q(self, observation, action, detach_model=False):
        return self.target_critic.get_q(observation, action, detach_model = detach_model)

    def get_log_prob(self, observation, action):
        if self.template is not None:
            observation = [self.template.replace("{obs}", obs) for obs in observation]
        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        action_ids = self.tokenizer(action, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        # action_embeds = self.model.get_input_embeddings()(action_ids["input_ids"]).detach()
        # obs_embeds = self.model.get_input_embeddings()(obs_ids["input_ids"]).detach()
        input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim = 1)
        # input_embeds = torch.cat([obs_embeds, action_embeds], dim = 1)
        attention_mask = torch.cat([obs_ids["attention_mask"], action_ids["attention_mask"]],\
                                dim = 1)
        outputs = self.model(input_ids=input_ids, attention_mask = attention_mask)
        values = None
        if isinstance(outputs, Tuple):
            values, outputs = outputs
        prediction_probs = self.softmax(outputs.logits)
        selected_prediction_probs = torch.take_along_dim(prediction_probs[:, obs_ids["attention_mask"].size(1)-1:-1],\
                                                 action_ids["input_ids"].unsqueeze(2), dim=2).squeeze(2)
        if values is not None:
            return values[:, obs_ids["attention_mask"].size(1)-1:-1], torch.log(selected_prediction_probs)*action_ids["attention_mask"], action_ids["attention_mask"]
        else:
            return torch.sum(torch.log(selected_prediction_probs)*action_ids["attention_mask"], dim = 1)

    def soft_update_target_critic(self, tau):
        # for target_critic, critic in zip(self.target_critics, self.critics):
        for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )
