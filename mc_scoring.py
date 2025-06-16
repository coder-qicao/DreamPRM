#!/usr/bin/env python3
"""
Generate PoT prefix dataset (train.json) and Monte Carlo meta rewards (meta.json).

python3 mc_scoring.py --raw_json data/test.json --train_json data/train_math.json --meta_json data/meta_math.json --openmath
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
    """
    Split OpenMathInstruct-1 solutions into logical reasoning steps.
    Each step represents a meaningful milestone in problem-solving.
    """
    import re
    
    steps = []
    
    # Pattern to find code blocks and outputs separately
    code_pattern = r'<llm-code>.*?</llm-code>'
    output_pattern = r'<llm-code-output>.*?</llm-code-output>'
    
    # Find all code blocks and outputs
    code_matches = list(re.finditer(code_pattern, sol, re.DOTALL))
    output_matches = list(re.finditer(output_pattern, sol, re.DOTALL))
    
    if not code_matches:
        # No code blocks found, return the whole solution
        return [sol.strip()] if sol.strip() else []
    
    current_pos = 0
    
    for i, code_match in enumerate(code_matches):
        # Step N: Add text before this code block (reasoning/setup)
        text_before = sol[current_pos:code_match.start()].strip()
        if text_before:
            steps.append(text_before)
        
        # Step N+1: Add the code block (implementation)
        steps.append(code_match.group().strip())
        
        # Step N+2: Add corresponding output (results)
        if i < len(output_matches):
            steps.append(output_matches[i].group().strip())
            current_pos = output_matches[i].end()
        else:
            current_pos = code_match.end()
    
    # Final step: Add any remaining text (conclusion)
    remaining_text = sol[current_pos:].strip()
    if remaining_text:
        steps.append(remaining_text)
    
    return [step for step in steps if step.strip()]


def build_prefix_dataset(raw_data: list[dict], openmath: bool=False) -> list[dict]:
    """
    Generate prefix entries for each partial solution in the new format.
    Returns list of {'id','sid','input','add','ground_truth'}.
    """
    prefixes = []
    for idx, item in enumerate(raw_data):
        sol = item.get('generated_solution', item.get('solution', ''))
        gt = str(item.get('expected_answer', item.get('answer', '')))
        
        # Get the original question/input
        original_input = item.get('question', item.get('input', f"Problem {idx}"))
        
        steps = split_steps_openmath(sol) if openmath else split_steps(sol)
        
        for sid, step in enumerate(steps, 1):
            # Build cumulative prefix (all steps up to current step)
            prefix_text = "\n".join(steps[:sid])
            
            prefixes.append({
                'id': idx,
                'sid': sid,
                'input': original_input,
                'add': prefix_text,
                'ground_truth': gt
            })
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
) -> tuple[float, int]:
    """
    Score a prefix by Monte Carlo sampling.
    Returns (accuracy, times_evaluated).
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
    
    accuracy = correct / M
    return accuracy, M


def main():
    parser = argparse.ArgumentParser(description="Generate PoT prefixes and Monte Carlo rewards")
    parser.add_argument('--raw_json',   required=True, help='Raw solutions JSON file')
    parser.add_argument('--train_json', required=True, help='Output prefix JSON (train.json)')
    parser.add_argument('--meta_json',  required=True, help='Output meta JSON (meta.json)')
    parser.add_argument('--model_name', default='/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/weights/Qwen2.5-Math-PRM-7B', help='HuggingFace model for scoring')
    parser.add_argument('--mc_samples', type=int, default=20, help='Number of Monte Carlo samples per prefix')
    parser.add_argument('--openmath',   action='store_true', help='Use OpenMathInstruct splitting')
    parser.add_argument('--skip_mc',    action='store_true', help='Skip Monte Carlo scoring (faster for testing)')
    args = parser.parse_args()

    # Load raw data
    with open(args.raw_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Build prefixes in new format
    prefixes = build_prefix_dataset(raw_data, openmath=args.openmath)
    
    # Save train.json (without scores)
    with open(args.train_json, 'w', encoding='utf-8') as f:
        json.dump(prefixes, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(prefixes)} prefixes to {args.train_json}")

    if args.skip_mc:
        print("Skipping Monte Carlo scoring (--skip_mc flag)")
        return

    # Initialize model/tokenizer for Monte Carlo scoring
    print("Loading model for Monte Carlo scoring...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    # Score prefixes and build meta.json
    print("Computing Monte Carlo scores...")
    metas = []
    
    for i, item in enumerate(prefixes):
        if i % 10 == 0:
            print(f"Processing {i}/{len(prefixes)}...")
            
        accuracy, times = monte_carlo_score_prefix(
            item['add'], 
            item['ground_truth'], 
            model, 
            tokenizer,
            M=args.mc_samples
        )
        
        # Convert accuracy to score (you can adjust this mapping)
        score = int(accuracy * 10)  # 0-10 scale like in examples
        
        meta_item = {
            'id': item['id'],
            'sid': item['sid'],
            'input': item['input'],
            'add': item['add'],
            'ground_truth': item['ground_truth'],
            'score': score,
            'times': times,
            'accuracy': accuracy
        }
        metas.append(meta_item)

    with open(args.meta_json, 'w', encoding='utf-8') as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(metas)} meta rewards to {args.meta_json}")


if __name__ == '__main__':
    main()