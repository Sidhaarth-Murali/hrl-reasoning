import torch
from tqdm import tqdm
MATH_ZERO_SHOT_TEMPLATE = """You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by
step. At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final
answer is $answer$. I hope it is correct."

{prompt}
"""

MATH_SELF_CORRECTION_TEMPLATE = """There might/might not be an error in the solution above because of lack of understanding of the question. Please correct
the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you
give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."
"""

def format_math_prompt(problem):
    """Format a math problem using the zero-shot template."""
    return MATH_ZERO_SHOT_TEMPLATE.format(prompt=problem)

def format_math_self_correction_prompt(initial_solution):
    """Format a math self-correction prompt with the initial solution."""
    return initial_solution + "\n\n" + MATH_SELF_CORRECTION_TEMPLATE

def generate_smart_correction_prompt(problem, solution, correction_model, tokenizer=None, device=None, batch_size=8):
    """Generate a custom correction prompt based on analyzing the problem and initial solution.
    
    Args:
        problem: The original math problem or a batch of problems (list)
        solution: The initial solution attempt or a batch of solutions (list)
        correction_model: Optional model to generate custom correction instructions
        tokenizer: Tokenizer for the correction model
        device: Device to run the model on
        batch_size: Number of items to process at once to avoid GPU memory issues
    
    Returns:
        A custom correction prompt or a list of prompts that address specific issues in the solutions
    """
    if correction_model is None:
        if isinstance(problem, list) and isinstance(solution, list):
            return [format_math_self_correction_prompt(p + s) for p, s in zip(problem, solution)]
        else:
            return format_math_self_correction_prompt(problem + solution)
    
    # Handle both single item and batch inputs
    is_batch = isinstance(problem, list) and isinstance(solution, list)
    
    if not is_batch:
        problem = [problem]
        solution = [solution]
    
    # Create analysis prompts for each problem-solution pair
    analysis_prompts = []
    for p, s in zip(problem, solution):
        analysis_prompt = f"""You are an expert math tutor reviewing a student's solution to a math problem.\n\n
        PROBLEM:{p}
        \n\n
        INITIAL SOLUTION:{s}
        \n\n
        PROMPT: First, analyze the solution for errors or misconceptions. Then, write a brief, helpful instruction that will guide the student toward correcting their solution.
        Your instruction should be specific to the errors you identified, but don't solve the problem for them.
        Your response should be ONLY the instruction for the student to improve their solution, nothing else. DO NOT include ANY SOLUTION.
        
        GUIDING INSTRUCTION:
        """
        analysis_prompts.append(analysis_prompt)
    
    try:
        # Ensure we're using the right device
        if device is None:
            device = correction_model.device

        # Process in batches to avoid GPU memory issues
        custom_instructions = []
        total_items = len(analysis_prompts)
        
        # Set up tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        
        # Process data in batches with progress bar
        for i in tqdm(range(0, total_items, batch_size), desc="Generating correction prompts"):
            # Get current batch
            batch_prompts = analysis_prompts[i:i+batch_size]
            
            # Tokenize current batch
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                truncation=True,
                padding=True,
            ).to(device)
            
            with torch.no_grad():
                # Generate outputs for current batch
                outputs = correction_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    use_cache=True,
                    num_beams=2, 
                )
                
            # Get input lengths for each item in batch to know where to start decoding from
            input_lengths = [len(ids) for ids in inputs["input_ids"]]
            
            # Decode outputs for current batch
            for j, output in enumerate(outputs):
                # Extract only the generated part (not the input)
                instruction = tokenizer.decode(output[input_lengths[j]:], skip_special_tokens=True).strip()
                custom_instructions.append(instruction)
                
            # Free up GPU memory
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Format the final prompts with the custom instructions
        final_prompts = [p + s + "\n\n" + "Suggestive Correction: \n\n" + instr 
                         for p, s, instr in zip(problem, solution, custom_instructions)]
        
        # Return single item or list depending on input type
        if not is_batch:
            return final_prompts[0]
        return final_prompts
        
    except Exception as e:
        print(f"Error generating custom correction prompts: {e}")
        # Fall back to static template if generation fails
        if not is_batch:
            return format_math_self_correction_prompt(problem[0] + solution[0])
        return [format_math_self_correction_prompt(p + s) for p, s in zip(problem, solution)]