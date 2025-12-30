import os
import pandas as pd
import torch
import argparse
from accelerate import Accelerator
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.notebook import tqdm
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from math_verify import LatexExtractionConfig, parse, verify
from peft import LoraConfig, get_peft_model

# Parse arguments
parser = argparse.ArgumentParser(description='GRPO training with optional two-turn mode')
parser.add_argument('--two-turn', action='store_true', default=False,
                    help='Enable two-turn mode: first turn -> feedback -> second turn -> GRPO on second turn rewards')
args = parser.parse_args()

TWO_TURN_MODE = args.two_turn

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Two-turn mode: {TWO_TURN_MODE}")
api_key = "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"
os.environ["OPENAI_API_KEY"] = api_key
# Constants
INITIAL_STR = "Solve the following math problem step-by-step by thinking deeply. After you find the final answer, express it in the format \\boxed{{your answer}}."
"Once your solution is complete, append exactly one EOS token (i.e. \n) to signal that you are done."

# Load the MATH dataset
file_path = "/home/pramit/hrl-nips-work/hrl-reasoning/dataset/MATH.csv"
math_df = pd.read_csv(file_path)

# Calculate the length of each correct answer
math_df['answer_length'] = math_df['correct_answer'].astype(str).apply(len)

# Filter to keep only the 5000 shortest answers
math_df = math_df.sort_values('answer_length').head(3500).reset_index(drop=True)
print(f"Filtered dataset to {len(math_df)} shortest answers")
math_df.head()
df = math_df



# Prepare dataset in the format expected by the GRPOTrainer
def prepare_math_dataset(df, max_samples=None):
    # Limit samples if specified
    if max_samples:
        df = df.head(max_samples)
    
    # Extract questions and answers
    questions = df['question'].tolist()
    answers = df['answer'].tolist()
    explanations = df['correct_answer'].tolist() if 'correct_answer' in df.columns else answers
    
    # Create prompt format
    prompts = [f"{INITIAL_STR}\n\n {q}\nSolution: \n\n" for q in questions]    
    # Create dataset dictionary
    dataset_dict = {
        "prompt": prompts,
        "explanation": explanations,
    }
    
    # Convert to HuggingFace dataset
    return Dataset.from_dict(dataset_dict)

# Create the dataset - limit to 1000 samples for faster training
math_dataset = prepare_math_dataset(df, max_samples=3500)
print(f"Prepared dataset with {len(math_dataset)} examples")

# Define model and training configuration
model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
training_args = GRPOConfig(
    output_dir="llama-3.2-1b-math-grpo", 
    logging_steps=10,
    learning_rate=1e-5,
    per_device_train_batch_size =32,
    remove_unused_columns=False,  
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    bf16=False,
    max_completion_length=512,
    num_generations=8,  
    save_strategy="steps",
    save_steps=50,
)

# Initialize tokenizer
print(f"Loading tokenizer from {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 
model.config.pad_token_id = tokenizer.eos_token_id 

# Global variables for two-turn mode (will be set after model initialization)
_model_for_twoturn = None
_tokenizer_for_twoturn = None

# Set global variables for two-turn mode
if TWO_TURN_MODE:
    _model_for_twoturn = model
    _tokenizer_for_twoturn = tokenizer
    print("Two-turn mode enabled: reward function will generate feedback and second-turn solutions")

from math_verify import LatexExtractionConfig, parse, verify

def generate_feedback(problem, first_turn_solution):
    """Generate feedback for the first turn solution."""
    feedback_prompt = f"""You are an expert math tutor reviewing a student's solution to a math problem.

PROBLEM: {problem}

INITIAL SOLUTION: {first_turn_solution}

PROMPT: First, analyze the solution for errors or misconceptions. Then, write a brief, helpful instruction that will guide the student toward correcting their solution. Your instruction should be specific to the errors you identified, but don't solve the problem for them. Your response should be ONLY the instruction for the student to improve their solution, nothing else. DO NOT include ANY SOLUTION.

GUIDING INSTRUCTION:"""
    
    if _model_for_twoturn is None or _tokenizer_for_twoturn is None:
        return "Please review your solution and correct any errors."
    
    _model_for_twoturn.eval()
    inputs = _tokenizer_for_twoturn(feedback_prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = _model_for_twoturn.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=_tokenizer_for_twoturn.eos_token_id
        )
    feedback = _tokenizer_for_twoturn.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return feedback

def build_correction_prompt(problem, first_turn_solution, feedback):
    """Build the correction prompt for second turn."""
    return f"""### Problem
{problem}

### Previous Solution
{first_turn_solution}

### Guidance
{feedback}

**Task — Follow the guidance above to correct your work. Think step-by-step and at the end of the Solution, when you give your final answer, write it in the form:**
**Final Answer: The final answer is $answer$. I hope it is correct.**
"""

def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs["explanation"]
    prompts = kwargs.get("prompt", [""] * len(completions))
    
    if TWO_TURN_MODE:
        # Two-turn mode: completions are first-turn solutions
        # Generate feedback and second-turn solutions
        second_turn_completions = []
        for prompt, first_turn_completion in zip(prompts, completions):
            # Extract problem from prompt
            problem = prompt.split("Solution:")[0].replace(INITIAL_STR, "").strip()
            
            # Generate feedback
            feedback = generate_feedback(problem, first_turn_completion)
            
            # Build correction prompt
            correction_prompt = build_correction_prompt(problem, first_turn_completion, feedback)
            
            # Generate second-turn solution
            _model_for_twoturn.eval()
            inputs = _tokenizer_for_twoturn(correction_prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                outputs = _model_for_twoturn.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=_tokenizer_for_twoturn.eos_token_id
                )
            second_turn = _tokenizer_for_twoturn.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            second_turn_completions.append(second_turn)
        
        # Compute rewards on second-turn completions
        completion_contents = second_turn_completions
    else:
        # Single-turn mode: use completions as-is
        completion_contents = [completion for completion in completions]
    
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    
    print(f"Average reward assigned to this group: {sum(rewards)/len(rewards)}")
    return rewards


lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Initialize GRPO Trainer
print(f"Setting up GRPO trainer with {model_name}")
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=math_dataset,
    reward_funcs=accuracy_reward)


accelerator = Accelerator()
print("Starting GRPO training...")
trainer.train()
# Save the model
final_model_path = "llama-3.2-1b-math-grpo-final"
trainer.save_model(final_model_path)
print(f"Training complete and model saved to {final_model_path}")

