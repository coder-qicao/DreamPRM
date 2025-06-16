from model import QwenVL_RM, QwenMath_RM
import random
import numpy as np
import torch
from time import sleep
from qwen_vl_utils import process_vision_info
import json

def set_cudnn(device="cuda"):
    torch.backends.cudnn.enabled = device == "cuda"
    torch.backends.cudnn.benchmark = device == "cuda"


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def stop_epoch(time=3):
    try:
        print("can break now")
        for i in range(time):
            sleep(1)
        print("wait for next epoch")
        return False
    except KeyboardInterrupt:
        return True


def compute_loss_accuracy(net, data_loader, criterion, device):
    net.eval()
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            total_loss += criterion(outputs, labels).item()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()

    return total_loss / (batch_idx + 1), correct / len(data_loader.dataset)


def load_QwenVL_RM(model_id = "reweighting/weights/base_model",
                      LN_id = "reweighting/weights/LN_weights.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = QwenVL_RM(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), model_id)
    best_model.LN.load_state_dict(torch.load(LN_id))
    best_model.to(device)
    return best_model


def load_QwenMath_RM(model_id = "reweighting/weights/base_model",
                        LN_id = "reweighting/weights/LN_weights.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = QwenMath_RM(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), model_id)
    best_model.LN.load_state_dict(torch.load(LN_id))
    best_model.to(device)
    return best_model


def generate_reward_model_input(input, response_step, image_path, processor):
    prompt = input + "\n\n" + response_step
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    return inputs


def generate_reward_model_input_math(input, response_step, processor):
    """Generate input for math reward model (text-only, no image)"""
    prompt = input + "\n\n" + response_step
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    return inputs


def create_dataset_mapping(json_file_path):
    """
    从JSON文件中提取所有唯一的dataset名称，并创建一个从0开始递增的数字映射字典
    支持视觉数据集和数学数据集两种格式

    参数:
    json_file_path: JSON文件路径

    返回:
    一个字典，格式为 {dataset_name1: 0, dataset_name2: 1, ...}
    """
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 检查数据集类型并提取唯一标识符
    unique_identifiers = set()
    
    # 检查是否为数学数据集（没有image_path但有id字段）
    if data and "image_path" not in data[0] and "id" in data[0]:
        # 数学数据集：使用problem_id作为域标识
        for item in data:
            if "id" in item:
                unique_identifiers.add(f"problem_{item['id']}")
    else:
        # 视觉数据集：使用dataset字段
        for item in data:
            if "dataset" in item:
                unique_identifiers.add(item["dataset"])

    # 创建映射字典（按字母排序）
    sorted_identifiers = sorted(list(unique_identifiers))
    mapping = {identifier: idx for idx, identifier in enumerate(sorted_identifiers)}

    return mapping