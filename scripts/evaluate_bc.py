import os, sys
import torch
import transformers
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from archer.environment import LLMBatchedMathEnv
from transformers import AutoModelForCausalLM, AutoTokenizer
from archer.utils import colorful_print
import numpy as np
import argparse
import random
import json

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate a trained BC model on mathematical problems')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the saved BC model checkpoint')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help='Name of the base model')
    parser.add_argument('--cache_dir', type=str, default='/home/pramit/hrl-nips-work/hrl-reasoning/.cache',
                        help='Cache directory for models')
    parser.add_argument('--num_problems', type=int, default=10,
                        help='Number of problems to evaluate')
    parser.add_argument('--output_file', type=str, default='bc_evaluation_results.json',
                        help='File to save evaluation results')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for generation')
    parser.add_argument('--max_tokens', type=int, default=100,
                        help='Maximum number of tokens to generate')
    args = parser.parse_args()

    # Check if model checkpoint exists
    if not os.path.exists(args.model_path):
        print(f"Model checkpoint not found at {args.model_path}")
        return

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Initialize math environment for test problems
    print("Initializing math environment...")
    env = LLMBatchedMathEnv(
        device=device,
        cache_dir=args.cache_dir,
        max_tokens=args.max_tokens
    )

    # Evaluate on random problems
    results = []
    correct_count = 0
    total_problems = min(args.num_problems, len(env.env_list[0].problems))
    
    print(f"Evaluating on {total_problems} problems...")
    problem_indices = random.sample(range(len(env.env_list[0].problems)), total_problems)
    
    for idx in tqdm(problem_indices):
        # Get problem and reference answer
        observations = env.reset(idx)
        problem = observations[0]
        reference_answer = env.env_list[0].curr_answer
        
        # Generate solution with the trained model
        inputs = tokenizer(problem, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=args.temperature
            )
        
        # Decode the generated solution
        solution = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Check if the solution is correct
        is_correct = env.env_list[0].is_correct(solution)
        if is_correct:
            correct_count += 1
        
        # Save the result
        results.append({
            "problem": problem,
            "reference_answer": reference_answer,
            "generated_solution": solution,
            "is_correct": is_correct
        })

    # Calculate accuracy
    accuracy = correct_count / total_problems
    print(f"Evaluation complete. Accuracy: {accuracy:.2f} ({correct_count}/{total_problems})")

    # Save results to file
    output_data = {
        "model_path": args.model_path,
        "model_name": args.model_name,
        "accuracy": accuracy,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "problems": results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main() 