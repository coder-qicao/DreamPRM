#!/usr/bin/env python3
"""
python3 basic_llm_judge.py --raw_json data/openmathinstruct-1.json --train_json data/train_math.json --meta_json data/meta_math.json
"""
import argparse
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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
        # Add text before code block (reasoning/setup)
        text_before = sol[current_pos:code_match.start()].strip()
        if text_before:
            steps.append(text_before)
        
        # Add code block (implementation)
        steps.append(code_match.group().strip())
        
        # Add corresponding output (results)
        if i < len(output_matches):
            steps.append(output_matches[i].group().strip())
            current_pos = output_matches[i].end()
        else:
            current_pos = code_match.end()
    
    # Add remaining text (conclusion)
    remaining_text = sol[current_pos:].strip()
    if remaining_text:
        steps.append(remaining_text)
    
    return [step for step in steps if step.strip()]


def build_prefix_dataset(raw_data: list[dict]) -> list[dict]:
    """Generate prefix entries for each partial solution."""
    prefixes = []
    for idx, item in enumerate(raw_data):
        question = item.get('question', '')
        expected_answer = str(item.get('expected_answer', ''))
        solution = item.get('generated_solution', '')
        
        steps = split_steps_openmath(solution)
        
        for sid, step in enumerate(steps, 1):
            prefix_text = "\n".join(steps[:sid])
            
            prefixes.append({
                'id': idx,
                'sid': sid,
                'input': question,
                'add': prefix_text,
                'ground_truth': expected_answer
            })
    return prefixes


class MathStepJudge:
    """LLM judge for scoring individual math solution steps"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-14B-Instruct"):
        print(f"Loading judge model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    def score_step(self, question: str, step: str, ground_truth: str) -> float:
        """Score a single step from 0.0 to 1.0"""
        prompt = f"""You are evaluating a step in a mathematical solution. Rate this step on a scale from 0.0 to 1.0 based on:
- Correctness of mathematical reasoning
- Helpfulness toward solving the problem
- Clarity and logical flow

Problem: {question}
Ground Truth Answer: {ground_truth}
Step to evaluate: {step}

Rate this step (0.0 = completely wrong/unhelpful, 1.0 = perfect):
Score:"""

        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Extract score from response
            score_match = re.search(r'(\d*\.?\d+)', response)
            if score_match:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))  # Clamp to [0,1]
            
        except Exception as e:
            print(f"Error scoring step: {e}")
        
        return 0.5  # Default score
    
    def batch_score_steps(self, items: list[dict]) -> list[float]:
        """Score multiple steps efficiently"""
        scores = []
        for i, item in enumerate(items):
            if i % 10 == 0:
                print(f"Scoring step {i+1}/{len(items)}")
            
            score = self.score_step(
                item['input'],
                item['add'], 
                item['ground_truth']
            )
            scores.append(score)
        
        return scores


def main():
    parser = argparse.ArgumentParser(description="LLM judge scoring for math problems")
    parser.add_argument('--raw_json', required=True, help='Raw OpenMathInstruct JSON file')
    parser.add_argument('--train_json', required=True, help='Output train JSON with LLM scores')
    parser.add_argument('--meta_json', required=True, help='Output meta JSON without scores')
    parser.add_argument('--judge_model', default='Qwen/Qwen2.5-14B-Instruct', help='LLM judge model')
    parser.add_argument('--skip_scoring', action='store_true', help='Skip LLM scoring for testing')
    args = parser.parse_args()

    # Load raw data
    print(f"Loading data from {args.raw_json}")
    with open(args.raw_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Build prefixes
    prefixes = build_prefix_dataset(raw_data)
    print(f"Generated {len(prefixes)} prefixes from {len(raw_data)} problems")

    # Save meta.json (clean prefixes for upper level)
    with open(args.meta_json, 'w', encoding='utf-8') as f:
        json.dump(prefixes, f, ensure_ascii=False, indent=2)
    print(f"Saved meta dataset to {args.meta_json}")

    if args.skip_scoring:
        # Save train.json without scores for testing
        with open(args.train_json, 'w', encoding='utf-8') as f:
            json.dump(prefixes, f, ensure_ascii=False, indent=2)
        print("Skipped LLM scoring")
        return

    # Initialize LLM judge
    judge = MathStepJudge(args.judge_model)

    # Score all steps
    print("Scoring steps with LLM judge...")
    scores = judge.batch_score_steps(prefixes)

    # Create train dataset with scores
    train_data = []
    for item, score in zip(prefixes, scores):
        train_item = {
            'id': item['id'],
            'sid': item['sid'],
            'input': item['input'],
            'add': item['add'],
            'ground_truth': item['ground_truth'],
            'score': int(score * 10),  # Convert to 0-10 scale
            'accuracy': score,        # Keep 0-1 scale
            'llm_judge_score': score  # Original LLM score
        }
        train_data.append(train_item)

    # Save train.json (with scores for lower level)
    with open(args.train_json, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"Saved train dataset to {args.train_json}")
    
    # Print statistics
    print(f"Score statistics:")
    print(f"  Mean score: {sum(scores)/len(scores):.3f}")
    print(f"  Min score: {min(scores):.3f}")
    print(f"  Max score: {max(scores):.3f}")
    print(f"  High scores (>0.8): {sum(1 for s in scores if s > 0.8)}")
    print(f"  Low scores (<0.3): {sum(1 for s in scores if s < 0.3)}")


if __name__ == '__main__':
    main()