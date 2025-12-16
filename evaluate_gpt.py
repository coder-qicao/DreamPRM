"""
评估脚本 - 使用 GPT Vision API 进行推理和评估
支持三种评估模式:
1. Zero-shot: 直接让 GPT 进行预测（无 CoT）
2. Best-of-N: 从多个 CoT 中选择最佳答案（基于一致性投票）
3. PRM-guided: 使用训练好的 PRM 对 CoT 进行评分并选择最佳答案
"""

import os
import json
import base64
import requests
import time
import argparse
from typing import List, Dict, Any
from collections import Counter

# ===== 配置参数 =====
parser = argparse.ArgumentParser(description="Evaluate using GPT Vision API")
parser.add_argument('--api_key_path', type=str, default='api.txt', help='Path to API key file')
parser.add_argument('--segmentation_json', type=str, default='segmentation.json', help='Path to segmentation JSON file')
parser.add_argument('--meta_json', type=str, default='data/medical_meta.json', help='Path to meta JSON file with CoTs')
parser.add_argument('--eval_mode', type=str, default='best_of_n',
                    choices=['zero_shot', 'best_of_n', 'prm_guided'],
                    help='Evaluation mode')
parser.add_argument('--model', type=str, default='gpt-4o', help='Model to use')
parser.add_argument('--base_url', type=str, default='https://api.apiyi.com/v1/chat/completions', help='API base URL')
parser.add_argument('--output_results', type=str, default='evaluation_results.json', help='Output results JSON')
parser.add_argument('--max_patients', type=int, default=None, help='Maximum number of patients to evaluate')

args = parser.parse_args()

# ===== 读取 API Key =====
with open(args.api_key_path, 'r', encoding='utf-8') as f:
    API_KEY = f.read().strip()

# ===== 辅助函数 =====
def image_to_base64(image_path: str) -> str:
    """将图片转换为 base64 编码"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_gpt_api(prompt: str, image_base64: str, model: str = 'gpt-4o') -> str:
    """调用 GPT Vision API"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}"
            }
        }
    ]

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 1000
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(args.base_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']

        except requests.exceptions.RequestException as e:
            print(f"      API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise

    return None

def extract_answer(text: str) -> str:
    """从文本中提取答案（yes/no）"""
    text_lower = text.lower()

    # 查找 "final answer:" 后面的内容
    if 'final answer:' in text_lower:
        answer_part = text_lower.split('final answer:')[-1].strip()
        if 'yes' in answer_part[:20]:  # 只检查前 20 个字符
            return 'yes'
        elif 'no' in answer_part[:20]:
            return 'no'

    # 如果没有找到 "final answer:"，检查整个文本
    # 统计 yes 和 no 的出现次数
    yes_count = text_lower.count(' yes')
    no_count = text_lower.count(' no')

    if yes_count > no_count:
        return 'yes'
    elif no_count > yes_count:
        return 'no'

    return 'unknown'

def create_zero_shot_prompt(task: str, image_description: str, image_diagnosis: str) -> str:
    """创建 zero-shot 提示（无 CoT）"""
    prompt = f"""
You are a medical imaging expert analyzing cervical cancer MRI scans.

Based on the provided image and text descriptions, please answer the following question:

Task: {task}

Image Description:
{image_description}

Image Diagnosis:
{image_diagnosis}

Please provide a clear yes/no answer in the format:

Final answer: [yes/no]
"""
    return prompt

# ===== 评估函数 =====
def evaluate_zero_shot():
    """Zero-shot 评估：直接让 GPT 预测，无 CoT"""
    print("=== Zero-shot 评估模式 ===\n")

    # 读取 segmentation 数据
    with open(args.segmentation_json, 'r', encoding='utf-8') as f:
        patients_data = json.load(f)

    if args.max_patients:
        patients_data = patients_data[:args.max_patients]

    # 定义任务
    tasks = [
        {
            'name': 'Lymphatic_node_transfer',
            'prompt': 'Determine if there is lymphatic node transfer (metastasis)'
        },
        {
            'name': 'Invade the uterus',
            'prompt': 'Determine if the tumor invades the uterus'
        },
        {
            'name': 'Invade the vagina',
            'prompt': 'Determine if the tumor invades the vagina'
        }
    ]

    results = []

    for patient_idx, patient in enumerate(patients_data):
        patient_id = patient['patient']
        print(f"评估患者 {patient_idx + 1}/{len(patients_data)}: {patient_id}")

        # 选择图片
        segmentation_images = patient.get('segmentation_images', [])
        if not segmentation_images:
            continue

        valid_image = None
        for img in segmentation_images:
            if img.endswith('.png') and os.path.exists(img):
                valid_image = img
                break

        if not valid_image:
            print(f"  警告: 没有找到有效图片")
            continue

        image_base64 = image_to_base64(valid_image)
        image_description = patient.get('Image description', '')
        image_diagnosis = patient.get('image diagnosis', '')

        # 对每个任务进行评估
        for task in tasks:
            task_name = task['name']
            task_prompt = task['prompt']
            ground_truth = patient.get(task_name, 'unknown').lower()

            print(f"  任务: {task_name} (Ground Truth: {ground_truth})")

            # 创建提示
            prompt = create_zero_shot_prompt(task_prompt, image_description, image_diagnosis)

            # 调用 API
            try:
                response = call_gpt_api(prompt, image_base64, model=args.model)
                predicted_answer = extract_answer(response)
                is_correct = (predicted_answer == ground_truth)

                print(f"    预测: {predicted_answer}, 正确: {is_correct}")

                results.append({
                    'patient': patient_id,
                    'task': task_name,
                    'ground_truth': ground_truth,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'response': response
                })

                time.sleep(1)  # 避免 API 限流

            except Exception as e:
                print(f"    错误: {e}")
                continue

    return results

def evaluate_best_of_n():
    """Best-of-N 评估：从多个 CoT 中选择最佳答案（基于投票）"""
    print("=== Best-of-N 评估模式 ===\n")

    # 读取 meta 数据（包含多个 CoT）
    with open(args.meta_json, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    print(f"读取 {len(meta_data)} 个 CoT 样本")

    # 按患者和任务分组
    patient_task_groups = {}
    for sample in meta_data:
        patient_id = sample['patient']
        task = sample['task']
        key = f"{patient_id}_{task}"

        if key not in patient_task_groups:
            patient_task_groups[key] = []

        patient_task_groups[key].append(sample)

    print(f"分组后有 {len(patient_task_groups)} 个 (患者, 任务) 组合\n")

    results = []

    for idx, (key, samples) in enumerate(patient_task_groups.items()):
        patient_id = samples[0]['patient']
        task = samples[0]['task']
        ground_truth = samples[0]['ground_truth'].lower()

        print(f"评估 {idx + 1}/{len(patient_task_groups)}: 患者 {patient_id}, 任务 {task}")
        print(f"  CoT 数量: {len(samples)}, Ground Truth: {ground_truth}")

        # 收集所有预测答案
        predictions = [sample['predicted_answer'] for sample in samples]

        # 投票选择最多的答案
        vote_counts = Counter(predictions)
        best_answer = vote_counts.most_common(1)[0][0]
        vote_ratio = vote_counts[best_answer] / len(predictions)

        is_correct = (best_answer == ground_truth)

        print(f"  预测分布: {dict(vote_counts)}")
        print(f"  最佳答案: {best_answer} (投票比例: {vote_ratio:.2f}), 正确: {is_correct}")

        results.append({
            'patient': patient_id,
            'task': task,
            'ground_truth': ground_truth,
            'num_cots': len(samples),
            'predictions': predictions,
            'vote_counts': dict(vote_counts),
            'best_answer': best_answer,
            'vote_ratio': vote_ratio,
            'is_correct': is_correct
        })

    return results

def evaluate_prm_guided():
    """PRM-guided 评估：使用训练好的 PRM 对 CoT 进行评分"""
    print("=== PRM-guided 评估模式 ===")
    print("注意：此模式需要先训练 PRM 模型")
    print("请先运行 main.py 训练模型，然后再使用此评估模式\n")

    # TODO: 实现 PRM 评分逻辑
    # 1. 加载训练好的 PRM 模型
    # 2. 对每个 CoT 进行评分
    # 3. 选择分数最高的 CoT

    return []

# ===== 主评估函数 =====
def run_evaluation():
    """运行评估"""
    print(f"评估模式: {args.eval_mode}")
    print(f"模型: {args.model}\n")

    # 根据评估模式选择评估函数
    if args.eval_mode == 'zero_shot':
        results = evaluate_zero_shot()
    elif args.eval_mode == 'best_of_n':
        results = evaluate_best_of_n()
    elif args.eval_mode == 'prm_guided':
        results = evaluate_prm_guided()
    else:
        raise ValueError(f"Unknown evaluation mode: {args.eval_mode}")

    # 保存结果
    print(f"\n保存评估结果到: {args.output_results}")
    with open(args.output_results, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 计算总体准确率
    if results:
        correct_count = sum(1 for r in results if r.get('is_correct', False))
        total_count = len(results)
        accuracy = correct_count / total_count * 100

        print("\n" + "="*60)
        print("评估结果汇总")
        print("="*60)
        print(f"总样本数: {total_count}")
        print(f"正确预测: {correct_count}")
        print(f"总体准确率: {accuracy:.2f}%")

        # 按任务统计准确率
        tasks = set(r['task'] for r in results)
        print("\n按任务统计:")
        for task in sorted(tasks):
            task_results = [r for r in results if r['task'] == task]
            task_correct = sum(1 for r in task_results if r.get('is_correct', False))
            task_total = len(task_results)
            task_accuracy = task_correct / task_total * 100 if task_total > 0 else 0
            print(f"  {task}: {task_correct}/{task_total} = {task_accuracy:.2f}%")

        print("="*60)

if __name__ == '__main__':
    print("=== 医学影像评估脚本 ===\n")
    run_evaluation()
    print("\n脚本执行完成!")
