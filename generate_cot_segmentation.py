"""
生成 CoT (Chain of Thought) 数据脚本 - 用于医学影像分析
使用 GPT-4o/o4-mini API 生成结构化的思维链推理

任务：预测三个二分类标签
1. Lymphatic_node_transfer (yes/no)
2. Invade the uterus (yes/no)
3. Invade the vagina (yes/no)
"""

import os
import json
import base64
import requests
import time
from pathlib import Path
from typing import List, Dict, Any
import argparse

# ===== 配置参数 =====
parser = argparse.ArgumentParser(description="Generate CoT for medical image analysis")
parser.add_argument('--api_key_path', type=str, default='api.txt', help='Path to API key file')
parser.add_argument('--segmentation_json', type=str, default='segmentation.json', help='Path to segmentation JSON file')
parser.add_argument('--output_train_json', type=str, default='data/medical_train.json', help='Output training data JSON')
parser.add_argument('--output_meta_json', type=str, default='data/medical_meta.json', help='Output meta data JSON')
parser.add_argument('--model', type=str, default='o4-mini', help='Model to use: o4-mini or gpt-4o')
parser.add_argument('--num_cots', type=int, default=8, help='Number of CoTs to generate per question')
parser.add_argument('--reasoning_effort', type=str, default='high', help='Reasoning effort for o4-mini: low, medium, high')
parser.add_argument('--base_url', type=str, default='https://api.apiyi.com/v1/chat/completions', help='API base URL')
parser.add_argument('--max_patients', type=int, default=None, help='Maximum number of patients to process (None for all)')

args = parser.parse_args()

# ===== 读取 API Key =====
with open(args.api_key_path, 'r', encoding='utf-8') as f:
    API_KEY = f.read().strip()

# ===== 辅助函数 =====
def image_to_base64(image_path: str) -> str:
    """将图片转换为 base64 编码"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def is_valid_image(image_path: str) -> bool:
    """检查图片是否有效（可以打开）"""
    if not os.path.exists(image_path):
        return False
    try:
        from PIL import Image
        img = Image.open(image_path)
        img.verify()
        return True
    except:
        return False

def select_valid_image(segmentation_images: List[str]) -> str:
    """从多个图片中选择第一个有效的 PNG 图片"""
    for img_path in segmentation_images:
        if img_path.endswith('.png') and is_valid_image(img_path):
            return img_path
    # 如果没有找到有效的PNG，返回第一个图片
    return segmentation_images[0] if segmentation_images else None

def create_structured_prompt(task: str, image_description: str, image_diagnosis: str) -> str:
    """
    创建结构化的 CoT 提示词

    Args:
        task: 任务描述（例如："Determine if there is lymphatic node transfer"）
        image_description: 图像描述文本
        image_diagnosis: 图像诊断文本
    """
    prompt = f"""
You are a medical imaging expert analyzing cervical cancer MRI scans.

You have been given medical imaging data (image + text descriptions) and a classification task.
Your task is to analyze the data by following exactly five steps:

Step 1: Restate the question.
  - Clearly rephrase or clarify the classification task in your own words.

Step 2: Gather evidence from the image and text descriptions.
  - Describe relevant visual and textual details that may help with the classification.
  - Focus on: lesion characteristics, anatomical structures, signal patterns, and invasion extent.

Step 3: Identify any necessary medical background knowledge.
  - List relevant medical facts about cervical cancer staging and invasion patterns.

Step 4: Reason using the available evidence.
  - Integrate the image findings, text descriptions, and medical knowledge to form a coherent reasoning path.
  - Consider: tumor size, location, signal characteristics, and relationship to adjacent structures.

Step 5: Summarize and conclude.
  - Provide a clear classification (yes or no), supported by the reasoning in previous steps.

Finally, report your answer in the following format:

Final answer: [yes/no]

--- Medical Data ---

Task: {task}

Image Description:
{image_description}

Image Diagnosis:
{image_diagnosis}

--- End of Medical Data ---

Please provide your step-by-step analysis following the 5-step structure above.
"""
    return prompt

def call_gpt_api(prompt: str, image_base64: str, model: str = 'gpt-4o',
                 reasoning_effort: str = 'high') -> Dict[str, Any]:
    """
    调用 GPT Vision API

    Args:
        prompt: 文本提示
        image_base64: base64 编码的图片
        model: 模型名称
        reasoning_effort: 推理力度（仅用于 o4-mini）

    Returns:
        API 响应字典
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # 构建消息内容
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}"
            }
        }
    ]

    # 构建 payload
    if model == 'o4-mini':
        # o4-mini 使用不同的 API 格式
        payload = {
            "model": model,
            "reasoning": {"effort": reasoning_effort},
            "input": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 2000
        }
    else:
        # gpt-4o 使用标准格式
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 2000
        }

    # 调用 API
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(args.base_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()

            # 提取响应文本
            if model == 'o4-mini':
                return {
                    'response': result.get('response', ''),
                    'reasoning': result.get('reasoning', '')
                }
            else:
                return {
                    'response': result['choices'][0]['message']['content']
                }

        except requests.exceptions.RequestException as e:
            print(f"API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
            else:
                raise

    return None

def extract_steps_from_cot(cot_text: str) -> List[Dict[str, str]]:
    """
    从 CoT 文本中提取各个步骤

    Returns:
        List of dicts with 'step_num' and 'content'
    """
    steps = []
    lines = cot_text.split('\n')
    current_step = None
    current_content = []

    for line in lines:
        # 检测步骤开始
        if line.strip().startswith('Step'):
            # 保存前一个步骤
            if current_step is not None:
                steps.append({
                    'step_num': current_step,
                    'content': '\n'.join(current_content).strip()
                })
            # 开始新步骤
            try:
                step_num = int(line.split(':')[0].replace('Step', '').strip())
                current_step = step_num
                current_content = [line]
            except:
                current_content.append(line)
        else:
            if current_step is not None:
                current_content.append(line)

    # 保存最后一个步骤
    if current_step is not None:
        steps.append({
            'step_num': current_step,
            'content': '\n'.join(current_content).strip()
        })

    return steps

def extract_final_answer(cot_text: str) -> str:
    """从 CoT 文本中提取最终答案"""
    lines = cot_text.lower().split('\n')
    for line in lines:
        if 'final answer:' in line:
            answer = line.split('final answer:')[-1].strip()
            if 'yes' in answer:
                return 'yes'
            elif 'no' in answer:
                return 'no'
    return 'unknown'

# ===== 主处理函数 =====
def process_segmentation_data():
    """处理 segmentation.json 数据并生成 CoT"""

    # 读取 segmentation.json
    print(f"读取数据文件: {args.segmentation_json}")
    with open(args.segmentation_json, 'r', encoding='utf-8') as f:
        patients_data = json.load(f)

    print(f"找到 {len(patients_data)} 个患者数据")

    # 限制处理的患者数量（用于测试）
    if args.max_patients:
        patients_data = patients_data[:args.max_patients]
        print(f"限制处理前 {args.max_patients} 个患者")

    # 定义三个任务
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

    train_data = []
    meta_data = []

    # 遍历每个患者
    for patient_idx, patient in enumerate(patients_data):
        patient_id = patient['patient']
        print(f"\n处理患者 {patient_idx + 1}/{len(patients_data)}: {patient_id}")

        # 选择有效的图片
        segmentation_images = patient.get('segmentation_images', [])
        if not segmentation_images:
            print(f"  警告: 患者 {patient_id} 没有 segmentation_images")
            continue

        valid_image_path = select_valid_image(segmentation_images)
        if not valid_image_path:
            print(f"  警告: 患者 {patient_id} 没有有效的图片")
            continue

        print(f"  选择图片: {valid_image_path}")

        # 转换图片为 base64
        try:
            image_base64 = image_to_base64(valid_image_path)
        except Exception as e:
            print(f"  错误: 无法读取图片 {valid_image_path}: {e}")
            continue

        # 获取文本描述
        image_description = patient.get('Image description', '')
        image_diagnosis = patient.get('image diagnosis', '')

        # 对于每个任务
        for task_idx, task in enumerate(tasks):
            task_name = task['name']
            task_prompt = task['prompt']
            ground_truth = patient.get(task_name, 'unknown')

            print(f"  任务 {task_idx + 1}/3: {task_name} (Ground Truth: {ground_truth})")

            # 创建结构化提示
            full_prompt = create_structured_prompt(
                task_prompt,
                image_description,
                image_diagnosis
            )

            # 生成多个 CoT（best-of-N）
            cot_responses = []
            for cot_idx in range(args.num_cots):
                print(f"    生成 CoT {cot_idx + 1}/{args.num_cots}...", end=' ')
                try:
                    response = call_gpt_api(
                        full_prompt,
                        image_base64,
                        model=args.model,
                        reasoning_effort=args.reasoning_effort
                    )

                    if response:
                        cot_text = response.get('response', '')
                        cot_responses.append({
                            'cot_text': cot_text,
                            'predicted_answer': extract_final_answer(cot_text)
                        })
                        print(f"完成 (预测: {cot_responses[-1]['predicted_answer']})")
                    else:
                        print("失败")

                    # 避免 API 限流
                    time.sleep(1)

                except Exception as e:
                    print(f"失败: {e}")
                    continue

            if not cot_responses:
                print(f"  警告: 任务 {task_name} 没有生成任何 CoT")
                continue

            # 处理每个 CoT 响应
            for cot_idx, cot_resp in enumerate(cot_responses):
                cot_text = cot_resp['cot_text']
                predicted_answer = cot_resp['predicted_answer']

                # 提取步骤
                steps = extract_steps_from_cot(cot_text)

                # 判断是否正确
                ground_truth_normalized = ground_truth.lower()
                is_correct = (predicted_answer == ground_truth_normalized)

                # 生成训练数据（每个步骤一个样本）
                for step in steps:
                    step_num = step['step_num']
                    step_content = step['content']

                    # 计算该步骤的准确率估计（正确答案为 1.0，错误答案递减）
                    if is_correct:
                        accuracy = 1.0 - (step_num - 1) * 0.05  # 后面的步骤略微降低
                    else:
                        accuracy = 0.3 - (step_num - 1) * 0.05  # 错误答案的步骤准确率低

                    accuracy = max(0.0, min(1.0, accuracy))  # 限制在 [0, 1]

                    train_sample = {
                        'id': f"{patient_id}_{task_name}_{cot_idx}_{step_num}",
                        'sid': step_num,
                        'input': full_prompt,
                        'add': step_content,
                        'ground_truth': ground_truth,
                        'image_path': valid_image_path,
                        'dataset': 'medical_segmentation',
                        'task': task_name,
                        'patient': patient_id,
                        'accuracy': accuracy,
                        'is_correct': is_correct
                    }
                    train_data.append(train_sample)

                # 生成 meta 数据（完整的 CoT）
                meta_sample = {
                    'id': f"{patient_id}_{task_name}_{cot_idx}",
                    'input': full_prompt + "\n\n" + cot_text,
                    'image_path': valid_image_path,
                    'true_false': is_correct,
                    'task': task_name,
                    'patient': patient_id,
                    'ground_truth': ground_truth,
                    'predicted_answer': predicted_answer
                }
                meta_data.append(meta_sample)

    # 保存训练数据
    os.makedirs(os.path.dirname(args.output_train_json), exist_ok=True)
    print(f"\n保存训练数据到: {args.output_train_json}")
    with open(args.output_train_json, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"训练数据样本数: {len(train_data)}")

    # 保存 meta 数据
    print(f"保存 meta 数据到: {args.output_meta_json}")
    with open(args.output_meta_json, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=2, ensure_ascii=False)
    print(f"Meta 数据样本数: {len(meta_data)}")

    # 打印统计信息
    print("\n=== 数据生成完成 ===")
    print(f"总患者数: {len(patients_data)}")
    print(f"总训练样本数: {len(train_data)}")
    print(f"总 meta 样本数: {len(meta_data)}")

    # 统计每个任务的准确率
    for task in tasks:
        task_name = task['name']
        task_meta = [m for m in meta_data if m['task'] == task_name]
        if task_meta:
            correct = sum(1 for m in task_meta if m['true_false'])
            total = len(task_meta)
            print(f"{task_name}: {correct}/{total} = {correct/total*100:.1f}% 正确")

if __name__ == '__main__':
    print("=== 医学影像 CoT 生成脚本 ===")
    print(f"模型: {args.model}")
    print(f"每个问题生成 CoT 数量: {args.num_cots}")
    print(f"推理力度: {args.reasoning_effort}")
    print()

    process_segmentation_data()

    print("\n脚本执行完成!")
