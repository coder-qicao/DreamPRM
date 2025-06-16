#!/usr/bin/env python3
"""
stage1_pipeline.py

Stage One: Generate PoT prefix dataset (train.json) and Monte Carlo meta rewards (meta.json).
This standalone script requires no modifications to data.py or model.py.
"""
import argparse
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def split_step(s_id: int, response: str) -> str:
    """Extract text for a single step labeled 'Step k'."""
    s = f"Step {s_id}"
    s_next = f"Step {s_id+1}"
    if s_next in response:
        part = response.split(s_next)[0]
    elif "Final answer" in response and s in response:
        part = response.split("Final answer")[0]
    else:
        return ""
    return part.strip()


def find_max_step(response: str) -> int:
    """Find the highest step number in the response."""
    nums = re.findall(r'Step[\s:]*(\d+)', response, flags=re.IGNORECASE)
    return max(map(int, nums)) if nums else 0


def split_steps(response: str) -> list[str]:
    """Split a full solution into a list of step texts."""
    max_step = find_max_step(response)
    return [split_step(i, response) for i in range(1, max_step+1) if split_step(i, response)]


def split_steps_openmath(sol: str) -> list[str]:
    """Split OpenMathInstruct-1 solutions by code/output blocks."""
    parts = re.split(r'<llm-code>', sol)[1:]
    steps = []
    for part in parts:
        code, rest = part.split('</llm-code>', 1)
        steps.append(code.strip())
        if '<llm-code-output>' in rest:
            output = rest.split('<llm-code-output>',1)[1].split('</llm-code-output>',1)[0]
            steps.append(output.strip())
    return steps


def build_prefix_dataset(raw_data: list[dict], openmath: bool=False) -> list[dict]:
    """
    Generate prefix entries for each partial solution.
    raw_data: list of dicts with keys 'generated_solution' and 'expected_answer'.
    Returns list of {'qid','prefix','ground_truth'}.
    """
    prefixes = []
    for idx, item in enumerate(raw_data):
        sol = item.get('generated_solution', item.get('solution', ''))
        gt  = str(item.get('expected_answer', item.get('answer', '')))
        steps = split_steps_openmath(sol) if openmath else split_steps(sol)
        for k in range(len(steps)):
            prefix_text = "\n".join(steps[:k+1])
            prefixes.append({'qid': str(idx), 'prefix': prefix_text, 'ground_truth': gt})
    return prefixes


def extract_final_answer(text: str) -> str:
    """Extract the final numeric answer from model output."""
    matches = re.findall(r'(-?\d+\.?\d*)', text)
    return matches[-1] if matches else text.strip()


def monte_carlo_score_prefix(
    prefix: str,
    ground_truth: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    M: int = 20,
    max_new_tokens: int = 64
) -> float:
    """
    Score a prefix by Monte Carlo sampling.
    Returns empirical accuracy = correct completions / M.
    """
    device = model.device
    enc = tokenizer(prefix, return_tensors='pt', truncation=True).to(device)
    correct = 0
    for _ in range(M):
        out = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = extract_final_answer(text)
        if pred == ground_truth:
            correct += 1
    return correct / M


def main():
    parser = argparse.ArgumentParser(description="Generate PoT prefixes and Monte Carlo rewards")
    parser.add_argument('--raw_json',   required=True, help='Raw solutions JSON file')
    parser.add_argument('--train_json', required=True, help='Output prefix JSON (train.json)')
    parser.add_argument('--meta_json',  required=True, help='Output meta JSON (meta.json)')
    parser.add_argument('--model_name', default='qwen2.5-math-rm', help='HuggingFace model for scoring')
    parser.add_argument('--mc_samples', type=int, default=20, help='Number of Monte Carlo samples per prefix')
    parser.add_argument('--openmath',   action='store_true', help='Use OpenMathInstruct splitting')
    args = parser.parse_args()

    # Load raw data
    with open(args.raw_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Build prefixes
    prefixes = build_prefix_dataset(raw_data, openmath=args.openmath)
    with open(args.train_json, 'w', encoding='utf-8') as f:
        json.dump(prefixes, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(prefixes)} prefixes to {args.train_json}")

    # Initialize model/tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    # Score prefixes
    metas = []
    for item in prefixes:
        score = monte_carlo_score_prefix(
            item['prefix'], item['ground_truth'], model, tokenizer,
            M=args.mc_samples
        )
        metas.append({'qid': item['qid'], 'prefix': item['prefix'], 'reward': score})
    with open(args.meta_json, 'w', encoding='utf-8') as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(metas)} meta rewards to {args.meta_json}")

if __name__ == '__main__':
    main()
