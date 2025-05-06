import os
import torch
import sys
import torch
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

# Replace 'your_file.pt' with the actual filename
file_path = '/home/pramit/hrl-nips-work/hrl-reasoning/.saved_models/score/trainer.pt'
# Path to saved model checkpoint
trainer_path = file_path
repo_id = 'SidhaarthMurali/flat-score-llama3.2-1b'

# Get your Hugging Face token (replace with your actual token or use environment variable)
import getpass
print("Enter your Hugging Face token:")
hf_token = getpass.getpass()

try:
    # Load the checkpoint
    print(f"Loading checkpoint from {trainer_path}")
    checkpoint = torch.load(trainer_path, map_location=torch.device('cpu'))
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    # Determine the base model name
    base_model = "meta-llama/Llama-3.2-1B-Instruct"  # Set this to your actual base model
    print(f"Using base model: {base_model}")
    
    # Log in to the Hugging Face Hub
    print("Logging in to Hugging Face Hub...")
    login(token=hf_token)
    
    # First, load the base model from HuggingFace
    print(f"Loading base model from {base_model}")
    base_model_obj = AutoModelForCausalLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Configure LoRA
    print("Configuring LoRA")
    lora_config = LoraConfig(
        r=16,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=32,
        lora_dropout=0.05
    )
    
    # Create a PEFT model
    print("Creating PEFT model with LoRA configuration")
    peft_model = get_peft_model(base_model_obj, lora_config)
    
    # Load the weights from the checkpoint
    if 'model_state_dict' in checkpoint:
        print("Loading model state dict from checkpoint")
        
        # Check if the state dict contains LoRA weights
        lora_params = [param for param in checkpoint['model_state_dict'].keys() if 'lora' in param.lower()]
        print(f"Found {len(lora_params)} LoRA parameters in the state dict")
        
        # Filter out only the LoRA parameters
        lora_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if 'lora' in k.lower()}
        
        # Load the LoRA weights into the PEFT model
        peft_model.load_state_dict(lora_state_dict, strict=False)
        
        # Merge the LoRA weights with the base model
        print("Merging LoRA weights with base model")
        merged_model = peft_model.merge_and_unload()
        
        # Configure generation settings
        merged_model.config.max_length = 2048
        
        # Save the merged model to HuggingFace Hub
        print(f"Pushing merged model to HuggingFace Hub as {repo_id}")
        merged_model.push_to_hub(repo_id, use_auth_token=hf_token)
        tokenizer.push_to_hub(repo_id, use_auth_token=hf_token)
        
        print(f"✓ Successfully pushed merged model to {repo_id}")
        print(f"You can now use the model with: model = AutoModelForCausalLM.from_pretrained('{repo_id}')")
    else:
        print("Error: Could not find model_state_dict in checkpoint")
        raise ValueError("No model state dict found in checkpoint")
    
except Exception as e:
    print(f"Error during model push: {str(e)}")
    import traceback
    traceback.print_exc()


breakpoint()


def safe_load_trajectories(file_path):
    """
    Safely load trajectories with helpful error handling
    """
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None
            
        print(f"Loading trajectories from {file_path}")
        trajectories = torch.load(file_path)
        print(f"Successfully loaded trajectories")
        return trajectories
        
    except Exception as e:
        print(f"Error loading trajectories: {str(e)}")
        return None

def analyze_trajectories(trajectories):
    """
    Perform basic analysis on the loaded trajectories
    """
    if trajectories is None:
        return
        
    print("\n=== Trajectories Analysis ===")
    
    if isinstance(trajectories, dict):
        print("Structure: Dictionary with keys:", list(trajectories.keys()))
        for key, value in trajectories.items():
            print(f"\nKey: {key}")
            print(f"Type: {type(value)}")
            if hasattr(value, "shape"):
                print(f"Shape: {value.shape}")
            elif isinstance(value, (list, tuple)):
                print(f"Length: {len(value)}")
                
    elif isinstance(trajectories, list):
        print(f"Structure: List with {len(trajectories)} items")
        if trajectories and len(trajectories) > 0:
            print(f"First item type: {type(trajectories[0])}")
            
    else:
        print(f"Structure: {type(trajectories)}")
        if hasattr(trajectories, "shape"):
            print(f"Shape: {trajectories.shape}")
            
    return trajectories

if __name__ == "__main__":
    # Use the provided path or default
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = '/home/pramit/hrl-nips-work/hrl-reasoning/.saved_models/trajectories.pt'
    
    trajectories = safe_load_trajectories(path)
    analyze_trajectories(trajectories)
    breakpoint()