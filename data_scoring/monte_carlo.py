#!/usr/bin/env python3
"""
Generate PoT prefix dataset (train.json) and Monte Carlo meta rewards (meta.json).

python3 monte_carlo.py --raw_json data/test.json --train_json data/train_math.json --meta_json data/meta_math.json --openmath
"""
import argparse
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def normalize_answer(answer):
    """Simple answer normalization for common math formats"""
    if not answer:
        return ""
    
    answer = str(answer).strip().lower()
    # Remove common formatting
    answer = re.sub(r'[,\s$%]', '', answer)
    answer = re.sub(r'\\.*?\{([^}]+)\}', r'\1', answer)  # Extract from \boxed{}
    answer = re.sub(r'^(answer|final answer|result)[:\s]*', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'[.!?]*$', '', answer)
    
    return answer


def are_answers_equivalent(ans1, ans2):
    """Check if two answers are equivalent"""
    norm1 = normalize_answer(ans1)
    norm2 = normalize_answer(ans2)
    
    if norm1 == norm2:
        return True
    
    # Try numeric comparison
    try:
        return abs(float(norm1) - float(norm2)) < 1e-6
    except:
        return False
    

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
    """Split OpenMathInstruct-1 solutions into logical reasoning steps."""
    steps = []
    code_pattern = r'<llm-code>.*?</llm-code>'
    output_pattern = r'<llm-code-output>.*?</llm-code-output>'
    
    code_matches = list(re.finditer(code_pattern, sol, re.DOTALL))
    output_matches = list(re.finditer(output_pattern, sol, re.DOTALL))
    
    if not code_matches:
        return [sol.strip()] if sol.strip() else []
    
    current_pos = 0
    for i, code_match in enumerate(code_matches):
        # Add text before code block
        text_before = sol[current_pos:code_match.start()].strip()
        if text_before:
            steps.append(text_before)
        
        # Add code block
        steps.append(code_match.group().strip())
        
        # Add corresponding output
        if i < len(output_matches):
            steps.append(output_matches[i].group().strip())
            current_pos = output_matches[i].end()
        else:
            current_pos = code_match.end()
    
    # Add remaining text
    remaining_text = sol[current_pos:].strip()
    if remaining_text:
        steps.append(remaining_text)
    
    return [step for step in steps if step.strip()]


def build_prefix_dataset(raw_data: list[dict], openmath: bool = False) -> list[dict]:
    """Generate prefix entries for each partial solution."""
    prefixes = []
    for idx, item in enumerate(raw_data):
        sol = item.get('generated_solution', item.get('solution', ''))
        gt = str(item.get('expected_answer', item.get('answer', '')))
        original_input = item.get('question', item.get('input', f"Problem {idx}"))
        dataset = item.get('dataset', 'unknown')
        
        steps = split_steps_openmath(sol) if openmath else split_steps(sol)
        
        for sid, step in enumerate(steps, 1):
            prefix_text = "\n".join(steps[:sid])
            prefixes.append({
                'id': idx,
                'sid': sid,
                'input': original_input,
                'add': prefix_text,
                'ground_truth': gt,
                'dataset': dataset
            })
    return prefixes


def build_meta_dataset(raw_data: list[dict]) -> list[dict]:
    """Generate meta entries in the new format."""
    meta_entries = []
    for idx, item in enumerate(raw_data):
        question = item.get('question', item.get('input', f"Problem {idx}"))
        solution = item.get('generated_solution', item.get('solution', ''))
        is_correct = item.get('is_correct', False)
        
        # Combine question and solution
        full_input = f"Question: {question}\n\nSolution: {solution}"
        
        meta_entries.append({
            'id': idx,
            'true_false': is_correct,
            'input': full_input
        })
    return meta_entries


def extract_final_answer(text: str) -> str:
    """Extract the final answer from model output."""
    patterns = [
        r'\\boxed\{([^}]+)\}',
        r'####\s*([^\n]+)',
        r'Final answer[:\s]*([^\n]+)',
        r'Answer[:\s]*([^\n]+)',
        r'([+-]?\d+\.?\d*)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    
    return text.strip()


def monte_carlo_score_prefix(prefix: str, ground_truth: str, model, tokenizer, M: int = 20) -> tuple[float, int]:
    """Score a prefix using Monte Carlo sampling."""
    device = model.device
    enc = tokenizer(prefix, return_tensors='pt', truncation=True).to(device)
    correct = 0
    
    for _ in range(M):
        try:
            out = model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                max_new_tokens=128,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            pred = extract_final_answer(text)
            
            if are_answers_equivalent(pred, ground_truth):
                correct += 1
        except:
            continue
    
    return correct / M, M


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo scoring for math problems")
    parser.add_argument('--raw_json', required=True, help='Raw solutions JSON file')
    parser.add_argument('--train_json', required=True, help='Output train JSON with MC scores')
    parser.add_argument('--meta_json', required=True, help='Output meta JSON without scores')
    parser.add_argument('--inference_model', default='/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/weights/Qwen3-8B', help='Model for inference')
    parser.add_argument('--mc_samples', type=int, default=20, help='Number of MC samples')
    parser.add_argument('--openmath', action='store_true', help='Use OpenMathInstruct splitting')
    parser.add_argument('--skip_mc', action='store_true', help='Skip MC scoring for testing')
    args = parser.parse_args()

    # Load raw data
    with open(args.raw_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Build meta dataset in new format
    meta_data = build_meta_dataset(raw_data)
    print(f"Generated {len(meta_data)} meta entries from {len(raw_data)} problems")

    # Save meta.json (new format with true/false labels)
    with open(args.meta_json, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)
    print(f"Saved meta dataset to {args.meta_json}")

    # Build prefixes for training
    prefixes = build_prefix_dataset(raw_data, openmath=args.openmath)
    print(f"Generated {len(prefixes)} prefixes from {len(raw_data)} problems")

    if args.skip_mc:
        # Save train.json without scores for testing
        with open(args.train_json, 'w', encoding='utf-8') as f:
            json.dump(prefixes, f, ensure_ascii=False, indent=2)
        print("Skipped MC scoring")
        return

    # Load model for Monte Carlo scoring
    print(f"Loading model: {args.inference_model}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.inference_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.inference_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Score prefixes
    print("Computing Monte Carlo scores...")
    train_data = []
    
    for i, item in enumerate(prefixes):
        if i % 10 == 0:
            print(f"Processing {i}/{len(prefixes)}")
            
        accuracy, times = monte_carlo_score_prefix(
            item['add'], 
            item['ground_truth'], 
            model, 
            tokenizer,
            M=args.mc_samples
        )
        
        train_item = {
            'id': item['id'],
            'sid': item['sid'],
            'input': item['input'],
            'add': item['add'],
            'ground_truth': item['ground_truth'],
            'dataset': item['dataset'],
            'score': int(accuracy * 10),
            'times': times,
            'accuracy': accuracy
        }
        train_data.append(train_item)

    # Save train.json (with MC scores for lower level)
    with open(args.train_json, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"Saved train dataset to {args.train_json}")
    
    # Print statistics
    accuracies = [item['accuracy'] for item in train_data]
    print(f"Mean accuracy: {sum(accuracies)/len(accuracies):.3f}")
    print(f"Min accuracy: {min(accuracies):.3f}")
    print(f"Max accuracy: {max(accuracies):.3f}")


if __name__ == '__main__':
    main()
