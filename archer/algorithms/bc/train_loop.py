from archer.data import DummyDataset
from archer.algorithms.bc import plain_bc_loss
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
import torch
import os

def train_loop(model,\
                bc_dataloader,\
                tokenizer,\
                optimizer=None,\
                iterations: int = 10,\
                grad_accum_steps: int = 1,\
                save_path: str = None,\
                use_wandb: bool = False,\
                **kwargs):
    """
    Simple behavior cloning training loop.
    
    Args:
        model: The language model to train
        bc_dataloader: DataLoader with training examples (observation, action pairs)
        tokenizer: Tokenizer for the model
        optimizer: Optimizer to use (will create one if not provided)
        iterations: Number of iterations through the dataset
        grad_accum_steps: Steps before optimizer step
        save_path: Path to save the model (directory)
        use_wandb: Whether to log to wandb
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    print_step = 0
    grad_step = 0
    
    for i in range(iterations):
        epoch_losses = []
        for batch in tqdm(bc_dataloader, desc=f"Iteration {i+1}/{iterations}"):
            loss = plain_bc_loss(model, tokenizer, **batch)
            epoch_losses.append(loss.item())
            
            grad_step += 1
            loss.backward()
            
            if grad_step % grad_accum_steps == 0:
                print_step += 1
                optimizer.step()
                optimizer.zero_grad()
                
                if print_step % 10 == 0:
                    print(f"Loss: {loss.item():.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Iteration {i+1}/{iterations}, Average Loss: {avg_loss:.4f}")
        
        if use_wandb:
            wandb.log({"bc_loss": avg_loss, "iteration": i})
    
    # Save the model
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        
        model_path = os.path.join(save_path, 'bc_model.pt')
        torch.save({
            'model_state_dict': model.cpu().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)
        print(f"Model saved to {model_path}")
    
    # Move model back to device
    device = next(model.parameters()).device
    model.to(device)
    
    return model