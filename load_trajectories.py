import os
import torch
import sys

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