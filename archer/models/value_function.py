import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class ValueFunction(nn.Module):
    """
    Memory-optimized value function for bilevel optimization in RL-Guided SCoRe.
    Estimates the expected future reward from the state at turn 2.
    """
    def __init__(self, model_name="distilroberta-base", device=None, cache_dir=None, use_gradient_checkpointing=True):
        super(ValueFunction, self).__init__()
        
        # Initialize the base model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'right'  # Keep beginning of problems
        
        # Load model with memory efficiency options
        self.base_model = AutoModel.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
        ).to(self.device)
        
        # Get the hidden size from the model config
        hidden_size = self.base_model.config.hidden_size
        
        # Simpler value head to save memory
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 1)
        ).to(self.device)
        
    def process_batch(self, problem_batch, initial_solution_batch, guidance_batch, revised_solution_batch, detach_base_model=False):
        """Process a single batch"""
        combined_input = [
            f"P:{p[:150]}|S1:{i[:150]}|G:{g[:100]}|S2:{r[:150]}" 
            for p, i, g, r in zip(
                problem_batch, 
                initial_solution_batch, 
                guidance_batch, 
                revised_solution_batch
            )
        ]
        
        # Tokenize with shorter max length
        inputs = self.tokenizer(
            combined_input,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Get the base model outputs
        if detach_base_model:
            with torch.no_grad():
                base_outputs = self.base_model(**inputs)
                # Use last hidden state instead of pooler to save memory
                pooled_output = base_outputs.last_hidden_state[:,0].detach()
        else:
            base_outputs = self.base_model(**inputs)
            pooled_output = base_outputs.last_hidden_state[:,0]
        
        # Get the value estimate
        return self.value_head(pooled_output)
        
    def forward(self, problem, initial_solution, guidance, revised_solution, detach_base_model=False):
        """Calculate the value estimate for the state after turn 2."""
        return self.process_batch(problem, initial_solution, guidance, revised_solution, detach_base_model)
    
    def get_value(self, problem, initial_solution, guidance, revised_solution, detach_base_model=False):
        """Convenience method to get just the value."""
        return self.forward(problem, initial_solution, guidance, revised_solution, detach_base_model)