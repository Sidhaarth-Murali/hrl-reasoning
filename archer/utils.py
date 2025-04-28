"""
Miscellaneous Utility Functions
"""
import click
import warnings
import re
from torch.utils.data import Dataset

def colorful_print(string: str, *args, **kwargs) -> None:
    print(click.style(string, *args, **kwargs))

def colorful_warning(string: str, *args, **kwargs) -> None:
    warnings.warn(click.style(string, *args, **kwargs))

import re

def clean_math_prompt(prompt: str) -> str:
    """
    Cleans a math‐expert prompt by:
      1. Extracting the problem statement (everything up to and including the first '?').
      2. Keeping the 'Suggestive Correction:' section, if present.
      3. Removing any 'CORRECTED INSTRUCTION:' block and everything after it.
      4. Prepending 'Problem: ' to the problem statement.
      5. Appending the guiding instruction and a 'Solution:' placeholder.
    """
    # 1. Extract problem (up to first '?')
    m = re.search(r'^(.*?\?)', prompt, flags=re.DOTALL)
    if m:
        problem = m.group(1).strip()
        rest = prompt[m.end():]
    else:
        problem = prompt.strip()
        rest = ''

    # 2. Extract the Suggestive Correction section
    sug_idx = rest.find('Suggestive Correction:')
    sug = rest[sug_idx:].strip() if sug_idx != -1 else ''

    # 3. Remove CORRECTED INSTRUCTION block
    sug = re.split(r'CORRECTED INSTRUCTION:', sug, maxsplit=1)[0].strip()

    # 4. Build the cleaned prompt
    cleaned = f"Problem: {problem}"
    if sug:
        cleaned += f"\n\n{sug}"

    # 5. Append guiding instruction and 'Solution:'
    guiding = (
        'You are a math expert. When you respond, respond only with the Solution of the final Problem, '
        'thinking step-by-step. Make use of the guiding instruction to refine or fix your solution if needed. '
        'At the end of the Solution, when you give your final answer, write it in the form '
        '"Final Answer: The final answer is $answer$. I hope it is correct."'
    )
    cleaned += f"\n\n{guiding}\n\nSolution:"

    return cleaned

