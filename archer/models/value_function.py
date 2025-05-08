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
            low_cpu_mem_usage=True,
        ).to(self.device)
        
        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
        
        # Get the hidden size from the model config
        hidden_size = self.base_model.config.hidden_size
        
        # Simpler value head to save memory
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 1)
        ).to(self.device)
        
    def process_batch(self, problem_batch, initial_solution_batch, guidance_batch, revised_solution_batch, detach_base_model=False):
        """Process a single batch to save memory"""
        # Create a more compact input representation
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
            max_length=256,  # Reduced to save memory
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
        """
        Calculate the value estimate for the state after turn 2.
        Memory-efficient implementation with batch processing.
        """
        # Process in smaller batches to save memory
        batch_size = len(problem)
        max_batch_size = 8  # Small batches to ensure we don't OOM
        
        # Initialize output tensor
        values = []
        
        for i in range(0, batch_size, max_batch_size):
            end_idx = min(i + max_batch_size, batch_size)
            batch_slice = slice(i, end_idx)
            
            # Process this small batch
            batch_values = self.process_batch(
                problem[batch_slice], 
                initial_solution[batch_slice], 
                guidance[batch_slice], 
                revised_solution[batch_slice],
                detach_base_model
            )
            values.append(batch_values)
            
            # Clear cache to save memory
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        
        # Combine all batch results
        return torch.cat(values, dim=0)
    
    def get_value(self, problem, initial_solution, guidance, revised_solution, detach_base_model=False):
        """Convenience method to get just the value."""
        return self.forward(problem, initial_solution, guidance, revised_solution, detach_base_model)