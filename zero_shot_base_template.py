import os
import json
import random
import base64
import requests
from pathlib import Path

# ===== 超参数配置 =====
API_KEY_PATH = r"api.txt"
DATASET_JSON_PATH = r"dataset.json"
TARGET_PATIENT = "sfybl_0"  # 指定要测试的patient名称
MODEL = "gpt-4o"
MAX_TOKENS = 2000
NUM_IMAGES = 5  # 随机选取的图片数量
BASE_URL = "https://api.apiyi.com/v1/chat/completions"

# ===== 读取API密钥 =====
with open(API_KEY_PATH, "r", encoding="utf-8") as f:
    api_key = f.read().strip()

# ===== 读取数据集JSON =====
with open(DATASET_JSON_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# ===== 查找目标patient =====
target_data = None
for item in dataset:
    if item["patient"] == TARGET_PATIENT:
        target_data = item
        break

if target_data is None:
    raise ValueError(f"未找到patient: {TARGET_PATIENT}")

print(f"找到目标patient: {TARGET_PATIENT}")
print(f"年龄: {target_data['age']}")

# ===== 获取T2A文件夹路径并随机选取5张图片 =====
t2a_folder = target_data["T2A"]
if not os.path.exists(t2a_folder):
    raise FileNotFoundError(f"T2A文件夹不存在: {t2a_folder}")

# 获取所有png图片
image_files = [f for f in os.listdir(t2a_folder) if f.lower().endswith('.png')]
if len(image_files) < NUM_IMAGES:
    print(f"警告: T2A文件夹中只有{len(image_files)}张图片，少于要求的{NUM_IMAGES}张")
    selected_images = image_files
else:
    selected_images = random.sample(image_files, NUM_IMAGES)

print(f"\n随机选取的{len(selected_images)}张图片:")
for img in selected_images:
    print(f"  - {img}")

# ===== 将图片转换为base64编码 =====
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

image_base64_list = []
for img_name in selected_images:
    img_path = os.path.join(t2a_folder, img_name)
    base64_str = encode_image_to_base64(img_path)
    image_base64_list.append(base64_str)

# ===== 构建消息内容 =====
# 从conversations中提取问题
question = ""
ground_truth = ""
for conv in target_data["conversations"]:
    if conv["from"] == "human":
        # 移除<image>标记
        question = conv["value"].replace("<image>", "").strip()
    elif conv["from"] == "gpt":
        ground_truth = conv["value"]

print(f"\n问题: {question}")
print(f"标准答案: {ground_truth}")

# 构建content数组
content = [{"type": "text", "text": question}]

# 添加所有图片（使用base64格式）
for base64_str in image_base64_list:
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_str}"
        }
    })

# ===== 调用API =====
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": content
        }
    ],
    "max_tokens": MAX_TOKENS
}

print("\n正在调用API...")
try:
    response = requests.post(BASE_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    
    result = response.json()
    api_response = result['choices'][0]['message']['content']
    
    print("\n" + "="*80)
    print("API响应:")
    print("="*80)
    print(api_response)
    print("="*80)
    
    print("\n" + "="*80)
    print("标准答案对比:")
    print("="*80)
    print(ground_truth)
    print("="*80)
    
except requests.exceptions.RequestException as e:
    print(f"API调用失败: {e}")
    if hasattr(e.response, 'text'):
        print(f"错误详情: {e.response.text}")

print("\n处理完成!")
