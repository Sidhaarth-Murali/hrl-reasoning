import torch
from tqdm import tqdm

CODE_ZERO_SHOT_TEMPLATE = """You are an expert programmer. Below is a programming problem. Write a solution in {language}.
Make sure your solution is correct, efficient, and addresses all the requirements of the problem.
When you're done, wrap your code in triple backticks with the language specified, like: ```{language} (your code here) ```

Problem:
{prompt}

Solution:
"""

CODE_SELF_CORRECTION_TEMPLATE = """Your code might have issues or bugs, or it may not be optimized. Please review your solution, identify any problems, and provide an improved solution.
Make sure your solution passes all test cases and meets all requirements. Remember to wrap your code in triple backticks with the language specified, like: ```{language} (your code here) ```
"""

def format_code_prompt(problem, language="python"):
    """Format a code problem using the zero-shot template."""
    return CODE_ZERO_SHOT_TEMPLATE.format(prompt=problem, language=language)

def format_code_self_correction_prompt(initial_solution, language="python"):
    """Format a code self-correction prompt with the initial solution."""
    return initial_solution + "\n\n" + CODE_SELF_CORRECTION_TEMPLATE.format(language=language)

def generate_smart_correction_prompt(problem, solution, correction_model=None, tokenizer=None, device=None, batch_size=8, language="python"):
    """Generate a custom correction prompt based on analyzing the problem and initial solution."""
    if correction_model is None:
        return format_code_self_correction_prompt(problem + solution, language)
    
    # Handle both single item and batch inputs
    is_batch = isinstance(problem, list) and isinstance(solution, list)
    
    if not is_batch:
        problem = [problem]
        solution = [solution]
    
    # Create analysis prompts for each problem-solution pair
    analysis_prompts = []
    for p, s in zip(problem, solution):
        analysis_prompt = f"""You are an expert programming mentor reviewing code written by a student.

PROBLEM:
{p}

STUDENT'S SOLUTION:
{s}

PROMPT: First, analyze the solution for bugs, inefficiencies, or edge cases it doesn't handle. Then, write a brief, helpful instruction that will guide the student toward correcting their solution.
Your instruction should be specific to the issues you identified, but don't solve the problem completely for them.
Your response should be ONLY the instruction for the student to improve their solution, nothing else. DO NOT write any code.

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
        final_prompts = [
            p + s + "\n\n" + f"Code Review Feedback: \n\n{instr}\n\nPlease fix these issues and provide an improved solution. Remember to wrap your code in triple backticks with the language specified, like: ```{language} (your code here) ```" 
            for p, s, instr in zip(problem, solution, custom_instructions)
        ]
        
        # Return single item or list depending on input type
        if not is_batch:
            return final_prompts[0]
        return final_prompts
        
    except Exception as e:
        print(f"Error generating custom correction prompts: {e}")
        # Fall back to static template if generation fails
        if not is_batch:
            return format_code_self_correction_prompt(problem[0] + solution[0], language)
        return [format_code_self_correction_prompt(p + s, language) for p, s in zip(problem, solution)] 