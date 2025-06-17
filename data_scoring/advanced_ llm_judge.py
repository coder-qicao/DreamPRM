#!/usr/bin/env python3
"""
python3 advanced_llm_judge.py --raw_json data/openmathinstruct-1.json --train_json data/train_math.json --meta_json data/meta_math.json
"""
import argparse
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict


def split_steps_openmath(sol: str) -> List[str]:
    """Split OpenMathInstruct-1 solutions into logical reasoning steps"""
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
    """Generate prefix entries for each partial solution"""
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
    """Build meta dataset with true_false labels and combined input"""
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


class AdvancedMathJudge:
    """Advanced LLM judge with specialized prompting for math problems"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-14B-Instruct"):
        print(f"Loading judge model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    def identify_step_type(self, step: str) -> str:
        """Identify the type of step for specialized evaluation"""
        step_lower = step.lower()
        
        if '<llm-code>' in step and '</llm-code>' in step:
            return 'code'
        elif '<llm-code-output>' in step and '</llm-code-output>' in step:
            return 'output'
        elif any(word in step_lower for word in ['thus', 'therefore', 'so', 'boxed']):
            return 'conclusion'
        elif any(word in step_lower for word in ['let', 'define', 'consider', 'assume']):
            return 'setup'
        else:
            return 'reasoning'
    
    def create_specialized_prompt(self, question: str, step: str, ground_truth: str, step_type: str) -> str:
        """Create specialized prompts based on step type"""
        
        base_instruction = f"""You are an expert mathematics teacher evaluating a student's solution step.

Problem: {question}
Correct Answer: {ground_truth}
Step to evaluate: {step}

"""
        
        if step_type == 'code':
            criteria = """Evaluate this code block based on:
1. Syntactic correctness and executability
2. Mathematical accuracy of the approach
3. Appropriateness for solving the problem
4. Clear variable names and logic"""
            
        elif step_type == 'output':
            criteria = """Evaluate this code output based on:
1. Numerical accuracy and precision
2. Consistency with the mathematical logic
3. Progress toward the final answer
4. Proper formatting and clarity"""
            
        elif step_type == 'conclusion':
            criteria = """Evaluate this concluding step based on:
1. Correctness of the final answer
2. Proper mathematical notation (e.g., \\boxed{})
3. Logical connection to previous steps
4. Completeness of the solution"""
            
        elif step_type == 'setup':
            criteria = """Evaluate this problem setup based on:
1. Correct interpretation of the problem
2. Appropriate variable definitions
3. Sound mathematical approach
4. Clear and logical organization"""
            
        else:  # reasoning
            criteria = """Evaluate this reasoning step based on:
1. Mathematical correctness and rigor
2. Logical flow and clarity
3. Relevance to solving the problem
4. Appropriate use of mathematical concepts"""
        
        scoring_guide = """
Rate this step on a scale from 0.0 to 1.0:
- 1.0: Excellent - Completely correct and very helpful
- 0.8: Good - Mostly correct with minor issues
- 0.6: Fair - Some correct elements but notable problems
- 0.4: Poor - Major errors but shows some understanding
- 0.2: Very Poor - Mostly incorrect but attempts the problem
- 0.0: Wrong - Completely incorrect or irrelevant

Provide only a single number between 0.0 and 1.0:"""
        
        return base_instruction + criteria + "\n\n" + scoring_guide
    
    def score_step(self, question: str, step: str, ground_truth: str) -> float:
        """Score a single step with specialized prompting"""
        step_type = self.identify_step_type(step)
        prompt = self.create_specialized_prompt(question, step, ground_truth, step_type)
        
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=15,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Extract score with multiple patterns
            score_patterns = [
                r'^(\d*\.?\d+)',  # Number at start
                r'(\d*\.?\d+)',   # Any number
            ]
            
            for pattern in score_patterns:
                score_match = re.search(pattern, response.strip())
                if score_match:
                    score = float(score_match.group(1))
                    return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"Error scoring step: {e}")
        
        return 0.5
    
    def batch_score_items(self, items: List[Dict]) -> List[Dict]:
        """Score multiple items and return training format"""
        train_data = []
        total = len(items)
        
        for i, item in enumerate(items):
            if i % 5 == 0:
                print(f"Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
            
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


def analyze_scores(scores: List[float]) -> Dict:
    """Analyze score distribution"""
    if not scores:
        return {}
    
    return {
        'mean': sum(scores) / len(scores),
        'min': min(scores),
        'max': max(scores),
        'high_quality': sum(1 for s in scores if s >= 0.8),
        'medium_quality': sum(1 for s in scores if 0.5 <= s < 0.8),
        'low_quality': sum(1 for s in scores if s < 0.5),
        'total': len(scores)
    }


def main():
    parser = argparse.ArgumentParser(description="Advanced LLM judge for math step scoring")
    parser.add_argument('--raw_json', required=True, help='Raw OpenMathInstruct JSON file')
    parser.add_argument('--train_json', required=True, help='Output train JSON with scores')
    parser.add_argument('--meta_json', required=True, help='Output meta JSON with true/false labels')
    parser.add_argument('--judge_model', default='Qwen/Qwen2.5-14B-Instruct', help='Judge model name')
    parser.add_argument('--max_items', type=int, help='Limit number of items for testing')
    parser.add_argument('--dataset_name', default='openmath', help='Dataset name to use')
    parser.add_argument('--accuracy_threshold', type=float, default=0.7, help='Threshold for meta true/false')
    parser.add_argument('--skip_scoring', action='store_true', help='Skip LLM scoring')
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
        # Score with advanced judge
        judge = AdvancedMathJudge(args.judge_model)
        print("Scoring steps with advanced LLM judge...")
        train_data = judge.batch_score_items(prefixes)

    # Build meta dataset
    meta_data = build_meta_dataset(train_data, args.accuracy_threshold)

    # Save results
    with open(args.train_json, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"Saved training dataset to {args.train_json}")
    
    with open(args.meta_json, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)
    print(f"Saved meta dataset to {args.meta_json}")
    
    # Print detailed statistics
    accuracies = [item['accuracy'] for item in train_data]
    true_count = sum(1 for item in meta_data if item['true_false'])
    stats = analyze_scores(accuracies)
    
    print(f"\nResults Summary:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Meta samples: {len(meta_data)}")
    print(f"  Mean accuracy: {stats['mean']:.3f}")
    print(f"  Score range: {stats['min']:.3f} - {stats['max']:.3f}")
    print(f"  High quality (≥0.8): {stats['high_quality']} ({stats['high_quality']/stats['total']*100:.1f}%)")
    print(f"  Medium quality (0.5-0.8): {stats['medium_quality']} ({stats['medium_quality']/stats['total']*100:.1f}%)")
    print(f"  Low quality (<0.5): {stats['low_quality']} ({stats['low_quality']/stats['total']*100:.1f}%)")
    print(f"  Meta true labels: {true_count}/{len(meta_data)} ({true_count/len(meta_data)*100:.1f}%)")


if __name__ == '__main__':
    main()
