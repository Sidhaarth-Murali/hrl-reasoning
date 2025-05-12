import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from datasets import load_dataset
from tqdm.notebook import tqdm

# ─────────── templates ───────────
ZERO_SHOT_TEMPLATE = """You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by
step. At the end of the Solution, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."

{problem}
"""

GUIDED_CORRECTION_TEMPLATE = """You are a math expert. Use the following guidance to refine your step-by-step solution:

Problem:
{problem}

Guidance:
{guidance}

Now provide the full corrected solution, step by step, ending with "Final Answer: The final answer is $answer$. I hope it is correct. 
Solution:
"""

# ─────────── helper functions ───────────
def format_zero_shot(problem: str) -> str:
    return ZERO_SHOT_TEMPLATE.format(problem=problem)

def build_analysis_prompts(problems, solutions):
    """Create prompts that ask the guider model to critique the solver’s initial solution."""
    prompts = []
    for p, s in zip(problems, solutions):
        prompts.append(
            f"""You are an expert math tutor reviewing a student's solution.

Problem:
{p}

Initial Solution:
{s}

Prompt: Identify any errors or misconceptions, then write a single, specific instruction to help the student correct it. Only output that instruction."""
        )
    return prompts

def generate_custom_guidance(prompts, model, tokenizer, device, batch_size=8):
    """
    Given analysis prompts, generate up to 512 tokens of guidance using the guider model.
    Returns a list of guidance strings.
    """
    all_guidance = []
    tokenizer.pad_token = tokenizer.eos_token

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating guidance"):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                num_beams=2,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_lens = [len(ids) for ids in inputs.input_ids]
        for j, out in enumerate(outs):
            guide = tokenizer.decode(out[input_lens[j]:], skip_special_tokens=True).strip()
            all_guidance.append(guide)

        torch.cuda.empty_cache()

    return all_guidance

# ─────────── device & model setup ───────────
device      = "cuda:0" if torch.cuda.is_available() else "cpu"
base_repo   = "meta-llama/Llama-3.2-1B-Instruct"
solver_repo = "SidhaarthMurali/grill-llama3.2-1b-f0.1v1-solver"
guider_repo = "SidhaarthMurali/grill-llama3.2-1b-f0.1v1-guider"

# Shared tokenizer (base tokenizer works for both fine‑tuned checkpoints)
tokenizer = AutoTokenizer.from_pretrained(base_repo)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load solver model (answers)
solver_model = AutoModelForCausalLM.from_pretrained(
    base_repo,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
)
solver_state = AutoModelForCausalLM.from_pretrained(solver_repo).state_dict()
solver_model.load_state_dict(solver_state)
solver_model.eval()

# Load guider model (critiques / guidance)
guider_model = AutoModelForCausalLM.from_pretrained(
    base_repo,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
)
guider_state = AutoModelForCausalLM.from_pretrained(guider_repo).state_dict()
guider_model.load_state_dict(guider_state)
guider_model.eval()

# ─────────── load data ───────────
ds            = load_dataset("HuggingFaceH4/MATH-500")
eval_dataset  = ds["test"]

batch_size = 32
results    = []

# ─────────── main loop ───────────
for i in tqdm(range(0, len(eval_dataset), batch_size), desc="Batches"):
    batch    = eval_dataset.select(range(i, min(i + batch_size, len(eval_dataset))))
    problems = [ex["problem"] for ex in batch]

    # ─── TURN 1: zero‑shot using solver_model ───
    prompts1 = [format_zero_shot(p) for p in problems]
    inputs1  = tokenizer(
        prompts1, return_tensors="pt",
        padding=True, truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        outs1 = solver_model.generate(
            **inputs1,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=2048,
            pad_token_id=tokenizer.eos_token_id,
        )

    turn1_sols = []
    for j, seq in enumerate(outs1):
        text   = tokenizer.decode(seq, skip_special_tokens=True)
        prompt = prompts1[j]
        sol    = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
        turn1_sols.append(sol)

    # ─── generate guidance with guider_model ───
    analysis_prompts = build_analysis_prompts(problems, turn1_sols)
    guidances = generate_custom_guidance(
        analysis_prompts,
        guider_model,          # <‑‑ use guider here
        tokenizer,
        device,
        batch_size=batch_size,
    )

    # ─── TURN 2: guided correction using solver_model ───
    prompts2 = [
        GUIDED_CORRECTION_TEMPLATE.format(
            problem=problems[j],
            guidance=guidances[j]
        )
        for j in range(len(problems))
    ]
    inputs2 = tokenizer(
        prompts2, return_tensors="pt",
        padding=True, truncation=True, max_length=1024
    ).to(device)

    with torch.no_grad():
        outs2 = solver_model.generate(
            **inputs2,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=2048,
            pad_token_id=tokenizer.eos_token_id,
        )

    turn2_sols = []
    for j, seq in enumerate(outs2):
        text   = tokenizer.decode(seq, skip_special_tokens=True)
        prompt = prompts2[j]
        sol    = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
        turn2_sols.append(sol)

    # ─── collect results ───
    for j, ex in enumerate(batch):
        results.append({
            "problem_id":       i + j,
            "problem":          ex["problem"],
            "reference_answer": ex["answer"],
            "solution_turn1":   turn1_sols[j],
            "guidance":         guidances[j],
            "solution_turn2":   turn2_sols[j],
        })

# ─────────── save to CSV ───────────
df = pd.DataFrame(results)
df.to_csv("math500_two_turn_with_guidance_f0.1.csv", index=False)
print(f"Saved {len(df)} examples to math500_two_turn_with_guidance.csv")
