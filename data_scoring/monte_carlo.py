#!/usr/bin/env python3
"""
python3 monte_carlo.py --raw_json data/openmathinstruct-1.json --train_json data/train_math.json --meta_json data/meta_math.json
"""
import argparse
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import os


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


def build_meta_dataset(prefixes: List[Dict], accuracy_threshold: float = 0.7) -> List[Dict]:
    """Build meta dataset with true_false labels and combined input."""
    meta_data = []
    
    for item in prefixes:
        # Combine question and partial response
        combined_input = f"Question: {item['input']}\n\nSolution: {item['add']}"
        
        # For meta dataset, we need a true/false label
        # We'll determine this based on Monte Carlo accuracy if available,
        # otherwise use a heuristic based on completion
        accuracy = item.get('accuracy', 0.5)  # Default if not available
        true_false = accuracy >= accuracy_threshold
        
        meta_item = {
            'id': item['id'],
            'true_false': true_false,
            'input': combined_input,
            'image_path': item['image_path']
        }
        meta_data.append(meta_item)
    
    return meta_data


class MonteCarloEvaluator:
    """Monte Carlo evaluator for math problem steps"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-14B-Instruct"):
        print(f"Loading evaluation model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    def extract_answer(self, text: str) -> str:
        """Extract the final numerical answer from model output"""
        # Look for boxed answers first
        boxed_pattern = r'\\boxed\{([^}]*)\}'
        match = re.search(boxed_pattern, text)
        if match:
            return match.group(1).strip()
        
        # Look for numerical answers at the end
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1]
        
        return ""
    
    def complete_solution(self, question: str, partial_solution: str) -> str:
        """Complete the solution from a partial state"""
        prompt = f"{question}\n\n{partial_solution}"
        
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return partial_solution + "\n" + response
            
        except Exception as e:
            print(f"Error in completion: {e}")
            return partial_solution
    
    def evaluate_step_mc(self, question: str, partial_solution: str, ground_truth: str, num_samples: int = 10) -> Tuple[int, float]:
        """Evaluate step using Monte Carlo sampling"""
        correct_count = 0
        
        for i in range(num_samples):
            if i % 5 == 0 and i > 0:
                print(f"  Sample {i}/{num_samples}")
            
            # Complete the solution
            completed = self.complete_solution(question, partial_solution)
            
            # Extract the final answer
            predicted_answer = self.extract_answer(completed)
            
            # Check if correct (handle different formats)
            if self.answers_match(predicted_answer, ground_truth):
                correct_count += 1
        
        accuracy = correct_count / num_samples
        return correct_count, accuracy
    
    def answers_match(self, pred: str, truth: str) -> bool:
        """Check if predicted answer matches ground truth"""
        if not pred or not truth:
            return False
        
        # Clean and normalize
        pred_clean = re.sub(r'[^\d.-]', '', pred.strip())
        truth_clean = re.sub(r'[^\d.-]', '', str(truth).strip())
        
        try:
            pred_num = float(pred_clean) if pred_clean else 0
            truth_num = float(truth_clean) if truth_clean else 0
            return abs(pred_num - truth_num) < 1e-6
        except:
            return pred_clean.lower() == truth_clean.lower()
    
    def batch_evaluate_mc(self, items: List[Dict], num_samples: int = 10) -> List[Dict]:
        """Evaluate multiple items with Monte Carlo"""
        results = []
        total = len(items)
        
        for i, item in enumerate(items):
            print(f"Evaluating item {i+1}/{total}")
            
            correct_count, accuracy = self.evaluate_step_mc(
                item['input'],
                item['add'],
                item['ground_truth'],
                num_samples
            )
            
            # Create training format result
            result = {
                'id': item['id'],
                'sid': item['sid'],
                'input': item['input'],
                'add': item['add'],
                'ground_truth': item['ground_truth'],
                'image_path': item['image_path'],
                'dataset': item['dataset'],
                'score': int(accuracy * 10),  # 0-10 scale
                'times': num_samples,
                'accuracy': accuracy  # 0-1 scale
            }
            results.append(result)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo scoring for math problems")
    parser.add_argument('--raw_json', required=True, help='Raw OpenMathInstruct JSON file')
    parser.add_argument('--train_json', required=True, help='Output train JSON with MC scores')
    parser.add_argument('--meta_json', required=True, help='Output meta JSON with true/false labels')
    parser.add_argument('--eval_model', default='Qwen/Qwen2.5-14B-Instruct', help='Model for evaluation')
    parser.add_argument('--num_samples', type=int, default=10, help='Monte Carlo samples per step')
    parser.add_argument('--max_items', type=int, help='Limit items for testing')
    parser.add_argument('--dataset_name', default='openmath', help='Dataset name to use')
    parser.add_argument('--accuracy_threshold', type=float, default=0.7, help='Threshold for meta true/false')
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

    # Monte Carlo evaluation
    evaluator = MonteCarloEvaluator(args.eval_model)
    print(f"Starting Monte Carlo evaluation with {args.num_samples} samples per step...")
    
    train_data = evaluator.batch_evaluate_mc(prefixes, args.num_samples)
    
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
