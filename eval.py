#!/usr/bin/env python3
"""
AIME Benchmark Evaluation Script
Evaluates trained reward models on AIME mathematical problems

python3 eval.py \
    --model_type qwenvl \
    --weights_path ./trained_weights \
    --aime_data ./aime_benchmark.json \
    --max_problems 50 \
    --device cuda:0 \
    --create_plots
"""

import argparse
import json
import torch
import numpy as np
import re
import os
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import your model classes and utilities
from model import QwenVL_RM, Llava_RM, QwenMath_RM
from data import QwenMathDataProcessor
from utils import load_model_config, load_trained_model, validate_model_config
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image, ImageDraw, ImageFont
import io


class AIMEEvaluator:
    def __init__(self, model_type, weights_path, device="cuda"):
        """
        Initialize AIME evaluator
        
        Args:
            model_type: Type of model ('qwenmath', 'qwenvl', 'llava')
            weights_path: Path to trained model weights
            device: Device to run evaluation on
        """
        self.model_type = model_type
        self.weights_path = weights_path
        self.device = device
        
        # Load model configuration
        try:
            self.config = load_model_config(weights_path)
            self.model_path = self.config['reward_model']
        except:
            print("Warning: Could not load config, using default model path")
            self.model_path = self._get_default_model_path()
        
        # Load model
        self.model = self._load_model()
        self.processor = self._load_processor()
        
        # Initialize results storage
        self.results = {
            'predictions': [],
            'ground_truths': [],
            'problem_ids': [],
            'scores': [],
            'processing_times': [],
            'step_evaluations': []
        }
    
    def _get_default_model_path(self):
        """Get default model path based on model type"""
        defaults = {
            'qwenmath': 'Qwen/Qwen2.5-Math-7B-Instruct',
            'qwenvl': 'Qwen/Qwen2-VL-2B-Instruct',
            'llava': 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf'
        }
        return defaults.get(self.model_type, defaults['qwenmath'])
    
    def _load_model(self):
        """Load the trained model"""
        print(f"Loading {self.model_type} model...")
        model = load_trained_model(
            model_type=self.model_type,
            device=self.device,
            model_path=self.model_path,
            weights_path=self.weights_path
        )
        return model
    
    def _load_processor(self):
        """Load appropriate processor/tokenizer"""
        if self.model_type == 'qwenmath':
            return AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                pad_token_id=151643
            )
        else:
            return AutoProcessor.from_pretrained(self.model_path)
    
    def load_aime_data(self, data_path):
        """
        Load AIME benchmark data
        
        Expected format:
        [
            {
                "problem_id": "aime_2023_1",
                "problem": "Find the number of...",
                "answer": 123,
                "year": 2023,
                "problem_number": 1,
                "difficulty": "medium"
            }
        ]
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} AIME problems")
        return data
    
    def create_dummy_image(self, text="AIME Problem"):
        """Create a dummy image for vision models when no image is provided"""
        img = Image.new('RGB', (512, 512), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to use a better font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Add text to image
        draw.text((50, 200), text, fill='black', font=font)
        draw.text((50, 250), "Math Problem", fill='black', font=font)
        
        return img
    
    def process_problem_for_model(self, problem_data):
        """Process AIME problem for specific model type"""
        problem_text = problem_data['problem']
        problem_id = problem_data.get('problem_id', 'unknown')
        
        if self.model_type == 'qwenmath':
            return self._process_for_qwenmath(problem_text)
        elif self.model_type == 'qwenvl':
            return self._process_for_qwenvl(problem_text, problem_id)
        elif self.model_type == 'llava':
            return self._process_for_llava(problem_text, problem_id)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _process_for_qwenmath(self, problem_text):
        """Process problem for QwenMath model"""
        # Create evaluation prompt
        eval_prompt = f"""<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant specialized in mathematics.<|im_end|>
<|im_start|>user
Evaluate this AIME problem solution approach:

Problem: {problem_text}

Rate the quality of approaching this problem (0.0 to 1.0):<|im_end|>
<|im_start|>assistant
"""
        
        # Tokenize
        inputs = self.processor(
            eval_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def _process_for_qwenvl(self, problem_text, problem_id):
        """Process problem for QwenVL model"""
        # Create dummy image since AIME is text-only
        dummy_image = self.create_dummy_image(f"AIME {problem_id}")
        
        # Save dummy image temporarily
        img_path = f"/tmp/aime_{problem_id}.png"
        dummy_image.save(img_path)
        
        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": f"Evaluate this AIME problem:\n\n{problem_text}\n\nRate the problem difficulty (0.0 to 1.0):"},
                ],
            }
        ]
        
        # Process with QwenVL processor
        from qwen_vl_utils import process_vision_info
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Clean up temporary image
        if os.path.exists(img_path):
            os.remove(img_path)
        
        return inputs
    
    def _process_for_llava(self, problem_text, problem_id):
        """Process problem for LLaVA model"""
        # Create dummy image
        dummy_image = self.create_dummy_image(f"AIME {problem_id}")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Evaluate this AIME problem:\n\n{problem_text}\n\nRate the problem difficulty (0.0 to 1.0):"},
                    {"type": "image"},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        
        inputs = self.processor(
            images=dummy_image, 
            text=text, 
            return_tensors='pt'
        ).to(self.device, torch.bfloat16)
        
        return inputs
    
    def evaluate_single_problem(self, problem_data):
        """Evaluate a single AIME problem"""
        import time
        
        start_time = time.time()
        
        try:
            # Process problem for model
            inputs = self.process_problem_for_model(problem_data)
            
            # Get model prediction
            with torch.no_grad():
                if self.model_type == 'qwenmath':
                    score = self.model(
                        inputs['input_ids'],
                        inputs['attention_mask']
                    )
                elif self.model_type == 'qwenvl':
                    score = self.model(
                        inputs['input_ids'].squeeze(),
                        inputs['attention_mask'].squeeze(),
                        inputs['pixel_values'].squeeze(),
                        inputs['image_grid_thw'].squeeze()
                    )
                elif self.model_type == 'llava':
                    score = self.model(
                        inputs['input_ids'].squeeze(),
                        inputs['attention_mask'].squeeze(),
                        inputs['pixel_values'].squeeze(),
                        inputs['image_sizes'].squeeze()
                    )
            
            processing_time = time.time() - start_time
            
            # Convert score to float
            if isinstance(score, torch.Tensor):
                score = score.item() if score.dim() == 0 else score[0].item()
            
            return {
                'score': score,
                'processing_time': processing_time,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'score': 0.0,
                'processing_time': processing_time,
                'success': False,
                'error': str(e)
            }
    
    def evaluate_dataset(self, aime_data, max_problems=None):
        """Evaluate the entire AIME dataset"""
        print(f"Evaluating {self.model_type} model on AIME benchmark...")
        
        if max_problems:
            aime_data = aime_data[:max_problems]
        
        results = []
        failed_problems = []
        
        for i, problem in enumerate(tqdm(aime_data, desc="Evaluating problems")):
            result = self.evaluate_single_problem(problem)
            
            # Store results
            results.append({
                'problem_id': problem.get('problem_id', f'problem_{i}'),
                'problem': problem['problem'],
                'ground_truth': problem.get('answer', None),
                'score': result['score'],
                'processing_time': result['processing_time'],
                'success': result['success'],
                'error': result['error'],
                'year': problem.get('year', None),
                'problem_number': problem.get('problem_number', None),
                'difficulty': problem.get('difficulty', 'unknown')
            })
            
            if not result['success']:
                failed_problems.append((i, result['error']))
        
        return results, failed_problems
    
    def calculate_metrics(self, results):
        """Calculate evaluation metrics"""
        scores = [r['score'] for r in results if r['success']]
        processing_times = [r['processing_time'] for r in results if r['success']]
        
        metrics = {
            'total_problems': len(results),
            'successful_evaluations': len(scores),
            'failed_evaluations': len(results) - len(scores),
            'success_rate': len(scores) / len(results) if results else 0,
            
            # Score statistics
            'mean_score': np.mean(scores) if scores else 0,
            'std_score': np.std(scores) if scores else 0,
            'min_score': np.min(scores) if scores else 0,
            'max_score': np.max(scores) if scores else 0,
            'median_score': np.median(scores) if scores else 0,
            
            # Performance statistics
            'mean_processing_time': np.mean(processing_times) if processing_times else 0,
            'total_processing_time': np.sum(processing_times) if processing_times else 0,
        }
        
        # Score distribution
        if scores:
            metrics['score_distribution'] = {
                'low_scores_0_0.3': len([s for s in scores if 0 <= s < 0.3]) / len(scores),
                'medium_scores_0.3_0.7': len([s for s in scores if 0.3 <= s < 0.7]) / len(scores),
                'high_scores_0.7_1.0': len([s for s in scores if 0.7 <= s <= 1.0]) / len(scores),
            }
        
        # Year-wise analysis if available
        year_scores = defaultdict(list)
        for r in results:
            if r['success'] and r['year']:
                year_scores[r['year']].append(r['score'])
        
        if year_scores:
            metrics['year_wise_performance'] = {
                year: {
                    'mean_score': np.mean(scores),
                    'count': len(scores)
                }
                for year, scores in year_scores.items()
            }
        
        # Difficulty-wise analysis if available
        difficulty_scores = defaultdict(list)
        for r in results:
            if r['success'] and r['difficulty'] != 'unknown':
                difficulty_scores[r['difficulty']].append(r['score'])
        
        if difficulty_scores:
            metrics['difficulty_wise_performance'] = {
                diff: {
                    'mean_score': np.mean(scores),
                    'count': len(scores)
                }
                for diff, scores in difficulty_scores.items()
            }
        
        return metrics
    
    def save_results(self, results, metrics, output_dir):
        """Save evaluation results and metrics"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(output_dir, f"aime_results_{self.model_type}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save metrics
        metrics_file = os.path.join(output_dir, f"aime_metrics_{self.model_type}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Save summary report
        self.generate_report(results, metrics, output_dir)
        
        print(f"Results saved to {output_dir}")
    
    def generate_report(self, results, metrics, output_dir):
        """Generate a comprehensive evaluation report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# AIME Benchmark Evaluation Report

**Model Type:** {self.model_type}
**Evaluation Date:** {timestamp}
**Model Path:** {self.model_path}
**Weights Path:** {self.weights_path}

## Summary Statistics

- **Total Problems:** {metrics['total_problems']}
- **Successful Evaluations:** {metrics['successful_evaluations']}
- **Success Rate:** {metrics['success_rate']:.2%}
- **Mean Score:** {metrics['mean_score']:.4f}
- **Standard Deviation:** {metrics['std_score']:.4f}
- **Score Range:** [{metrics['min_score']:.4f}, {metrics['max_score']:.4f}]
- **Median Score:** {metrics['median_score']:.4f}

## Performance Metrics

- **Mean Processing Time:** {metrics['mean_processing_time']:.3f} seconds
- **Total Processing Time:** {metrics['total_processing_time']:.2f} seconds

## Score Distribution

"""
        
        if 'score_distribution' in metrics:
            dist = metrics['score_distribution']
            report += f"""
- **Low Scores (0.0-0.3):** {dist['low_scores_0_0.3']:.2%}
- **Medium Scores (0.3-0.7):** {dist['medium_scores_0.3_0.7']:.2%}
- **High Scores (0.7-1.0):** {dist['high_scores_0.7_1.0']:.2%}
"""
        
        # Year-wise performance
        if 'year_wise_performance' in metrics:
            report += "\n## Year-wise Performance\n\n"
            for year, perf in metrics['year_wise_performance'].items():
                report += f"- **{year}:** {perf['mean_score']:.4f} (n={perf['count']})\n"
        
        # Difficulty-wise performance
        if 'difficulty_wise_performance' in metrics:
            report += "\n## Difficulty-wise Performance\n\n"
            for diff, perf in metrics['difficulty_wise_performance'].items():
                report += f"- **{diff.title()}:** {perf['mean_score']:.4f} (n={perf['count']})\n"
        
        # Failed evaluations
        failed_count = metrics['total_problems'] - metrics['successful_evaluations']
        if failed_count > 0:
            report += f"\n## Failed Evaluations\n\n{failed_count} problems failed to evaluate.\n"
        
        # Top and bottom performers
        successful_results = [r for r in results if r['success']]
        if successful_results:
            sorted_results = sorted(successful_results, key=lambda x: x['score'], reverse=True)
            
            report += "\n## Top 5 Highest Scoring Problems\n\n"
            for i, result in enumerate(sorted_results[:5], 1):
                report += f"{i}. **{result['problem_id']}** - Score: {result['score']:.4f}\n"
            
            report += "\n## Top 5 Lowest Scoring Problems\n\n"
            for i, result in enumerate(sorted_results[-5:], 1):
                report += f"{i}. **{result['problem_id']}** - Score: {result['score']:.4f}\n"
        
        # Save report
        report_file = os.path.join(output_dir, f"aime_report_{self.model_type}.md")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {report_file}")
    
    def create_visualizations(self, results, metrics, output_dir):
        """Create visualization plots"""
        scores = [r['score'] for r in results if r['success']]
        
        if not scores:
            print("No successful evaluations to visualize")
            return
        
        plt.style.use('default')
        
        # Score distribution histogram
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Reward Score')
        plt.ylabel('Frequency')
        plt.title(f'AIME Score Distribution - {self.model_type.upper()}')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'score_distribution_{self.model_type}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Year-wise performance if available
        if 'year_wise_performance' in metrics:
            years = list(metrics['year_wise_performance'].keys())
            year_scores = [metrics['year_wise_performance'][year]['mean_score'] for year in years]
            
            plt.figure(figsize=(10, 6))
            plt.plot(years, year_scores, marker='o', linewidth=2, markersize=8)
            plt.xlabel('Year')
            plt.ylabel('Mean Score')
            plt.title(f'AIME Performance by Year - {self.model_type.upper()}')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f'year_performance_{self.model_type}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on AIME benchmark")
    parser.add_argument("--model_type", type=str, required=True, 
                       choices=["qwenmath", "qwenvl", "llava"],
                       help="Type of model to evaluate")
    parser.add_argument("--weights_path", type=str, required=True,
                       help="Path to trained model weights")
    parser.add_argument("--aime_data", type=str, required=True,
                       help="Path to AIME benchmark data (JSON file)")
    parser.add_argument("--output_dir", type=str, default="./aime_evaluation",
                       help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run evaluation on")
    parser.add_argument("--max_problems", type=int, default=None,
                       help="Maximum number of problems to evaluate (for testing)")
    parser.add_argument("--create_plots", action="store_true",
                       help="Create visualization plots")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AIME Benchmark Evaluation")
    print("=" * 60)
    print(f"Model Type: {args.model_type}")
    print(f"Weights Path: {args.weights_path}")
    print(f"AIME Data: {args.aime_data}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Device: {args.device}")
    
    # Initialize evaluator
    evaluator = AIMEEvaluator(
        model_type=args.model_type,
        weights_path=args.weights_path,
        device=args.device
    )
    
    # Load AIME data
    aime_data = evaluator.load_aime_data(args.aime_data)
    
    # Run evaluation
    results, failed_problems = evaluator.evaluate_dataset(
        aime_data, 
        max_problems=args.max_problems
    )
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Problems: {metrics['total_problems']}")
    print(f"Successful Evaluations: {metrics['successful_evaluations']}")
    print(f"Success Rate: {metrics['success_rate']:.2%}")
    print(f"Mean Score: {metrics['mean_score']:.4f} ± {metrics['std_score']:.4f}")
    print(f"Score Range: [{metrics['min_score']:.4f}, {metrics['max_score']:.4f}]")
    print(f"Mean Processing Time: {metrics['mean_processing_time']:.3f} seconds")
    
    if failed_problems:
        print(f"\nFailed Problems: {len(failed_problems)}")
        for i, error in failed_problems[:3]:  # Show first 3 errors
            print(f"  Problem {i}: {error}")
    
    # Save results
    evaluator.save_results(results, metrics, args.output_dir)
    
    # Create visualizations if requested
    if args.create_plots:
        evaluator.create_visualizations(results, metrics, args.output_dir)
    
    print(f"\nEvaluation completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()