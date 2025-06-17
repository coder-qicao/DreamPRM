#!/usr/bin/env python3
"""
python3 llm_judge_scoring.py --raw_json data/openmathinstruct-1.json --train_json data/train_math.json --meta_json data/meta_math.json
"""
import argparse
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict


def split_steps_openmath(sol: str) -> List[str]:
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
        text_before = sol[current_pos:code_match.start()].strip()
        if text_before:
            steps.append(text_before)
        
        steps.append(code_match.group().strip())
        
        if i < len(output_matches):
            steps.append(output_matches[i].group().strip())
            current_pos = output_matches[i].end()
        else:
            current_pos = code_match.end()
    
    remaining_text = sol[current_pos:].strip()
    if remaining_text:
        steps.append(remaining_text)
    
    return [step for step in steps if step.strip()]


def build_prefix_dataset(raw_data: List[Dict], default_dataset: str = "openmath") -> List[Dict]:
    """Generate prefix entries for each partial solution."""
    prefixes = []
    for idx, item in enumerate(raw_data):
        question = item.get('question', '')
        expected_answer = str(item.get('expected_answer', ''))
        solution = item.get('generated_solution', '')
        
        # Extract dataset info if available, otherwise use default
        dataset_name = item.get('dataset', default_dataset)
        
        steps = split_steps_openmath(solution)
        
        for sid, step in enumerate(steps, 1):
            prefix_text = "\n".join(steps[:sid])
            
            prefixes.append({
                'id': idx,
                'sid': sid,
                'input': question,
                'add': prefix_text,
                'ground_truth': expected_answer,
                'image_path': item.get('image_path', ''),  # Empty if no image
                'dataset': dataset_name
            })
    return prefixes


def build_meta_dataset(train_data: List[Dict], accuracy_threshold: float = 0.7) -> List[Dict]:
    """Build meta dataset with true_false labels and combined input."""
    meta_data = []
    
    for item in train_data:
        # Combine question and partial response
        combined_input = f"Question: {item['input']}\n\nSolution: {item['add']}"
        
        # Determine true/false based on accuracy
        accuracy = item.get('accuracy', 0.5)
        true_false = accuracy >= accuracy_threshold
        
        meta_item = {
            'id': item['id'],
            'true_false': true_false,
            'input': combined_input,
            'image_path': item['image_path']
        }
        meta_data.append(meta_item)
    
    return meta_data


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
    
    def batch_score_steps(self, items: List[Dict]) -> List[Dict]:
        """Score multiple steps and return training format"""
        train_data = []
        total = len(items)
        
        for i, item in enumerate(items):
            if i % 10 == 0:
                print(f"Scoring step {i+1}/{total}")
            
            score = self.score_step(
                item['input'],
                item['add'], 
                item['ground_truth']
            )
            
            # Create training format result
            train_item = {
                'id': item['id'],
                'sid': item['sid'],
                'input': item['input'],
                'add': item['add'],
                'ground_truth': item['ground_truth'],
                'image_path': item['image_path'],
                'dataset': item['dataset'],
                'score': int(score * 10),  # 0-10 scale
                'times': 1,  # LLM judge uses single evaluation
                'accuracy': score  # 0-1 scale
            }
            train_data.append(train_item)
        
        return train_data


def main():
    parser = argparse.ArgumentParser(description="LLM judge scoring for math problems")
    parser.add_argument('--raw_json', required=True, help='Raw OpenMathInstruct JSON file')
    parser.add_argument('--train_json', required=True, help='Output train JSON with LLM scores')
    parser.add_argument('--meta_json', required=True, help='Output meta JSON with true/false labels')
    parser.add_argument('--judge_model', default='Qwen/Qwen2.5-14B-Instruct', help='LLM judge model')
    parser.add_argument('--max_items', type=int, help='Limit number of items for testing')
    parser.add_argument('--dataset_name', default='openmath', help='Dataset name to use')
    parser.add_argument('--accuracy_threshold', type=float, default=0.7, help='Threshold for meta true/false')
    parser.add_argument('--skip_scoring', action='store_true', help='Skip LLM scoring for testing')
    args = parser.parse_args()

    # Load and process data
    print(f"Loading data from {args.raw_json}")
    with open(args.raw_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    if args.max_items:
        raw_data = raw_data[:args.max_items]
        print(f"Limited to {args.max_items} items for testing")
    
    # Build prefixes
    prefixes = build_prefix_dataset(raw_data, args.dataset_name)
    print(f"Generated {len(prefixes)} prefixes from {len(raw_data)} problems")

    if args.skip_scoring:
        # Create dummy training data for testing
        train_data = []
        for item in prefixes:
            train_item = {
                'id': item['id'],
                'sid': item['sid'],
                'input': item['input'],
                'add': item['add'],
                'ground_truth': item['ground_truth'],
                'image_path': item['image_path'],
                'dataset': item['dataset'],
                'score': 5,  # Default score
                'times': 1,
                'accuracy': 0.5  # Default accuracy
            }
            train_data.append(train_item)
        print("Skipped LLM scoring - using default scores")
    else:
        # Initialize LLM judge and score
        judge = MathStepJudge(args.judge_model)
        print("Scoring steps with LLM judge...")
        train_data = judge.batch_score_steps(prefixes)

    # Build meta dataset
    meta_data = build_meta_dataset(train_data, args.accuracy_threshold)

    # Save results
    with open(args.train_json, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"Saved training dataset to {args.train_json}")
    
    with open(args.meta_json, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)
    print(f"Saved meta dataset to {args.meta_json}")
    
    # Print statistics
    accuracies = [item['accuracy'] for item in train_data]
    true_count = sum(1 for item in meta_data if item['true_false'])
    
    print(f"\nResults Summary:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Meta samples: {len(meta_data)}")
    print(f"  Mean accuracy: {sum(accuracies)/len(accuracies):.3f}")
    print(f"  Min accuracy: {min(accuracies):.3f}")
    print(f"  Max accuracy: {max(accuracies):.3f}")
    print(f"  High quality steps (>0.8): {sum(1 for a in accuracies if a > 0.8)}")
    print(f"  Meta true labels: {true_count}/{len(meta_data)} ({true_count/len(meta_data)*100:.1f}%)")


if __name__ == '__main__':
    main()
