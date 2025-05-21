# Written by QI CAO on May 21, 2025.
# All code is original unless otherwise noted.

import sys
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='Project root path')
parser.add_argument('--gpu', type=str, default='0', help='GPU device ID (CUDA_VISIBLE_DEVICES)')
args = parser.parse_args()

# Append project root path to sys.path for module importing
sys.path.append(args.path)

# Set visible CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Change working directory to the project root
os.chdir(args.path)

# Optional: Print confirmation
print(f"CUDA_VISIBLE_DEVICES set to {args.gpu}")
print(f"Working directory changed to {args.path}")

# main
from utils.json_processor import read_json, write_json
import random

dataset = "MMPR"
path = "MC/results/MMPR/Tree/0.json"
dataset_list = ["ai2d_train_12k_en_20240410_extracted_pairs_vqa_correctness_rules.jsonl",
               "chartqa_trainval_30k_w_csv_en_20240402_extracted_pairs_vqa_correctness_rules.jsonl",
               "m3cot_train_extracted_pairs_vqa_correctness_rules.jsonl",
               "scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_correctness_rules.jsonl",
               "mapqa_suv_en_20240402_extracted_pairs_vqa_correctness_rules.jsonl",
               "geo170k_extracted_full_pairs_vqa_correctness_rules.jsonl",
               "CLEVR_math_en_20240402_extracted_pairs_vqa_correctness_rules.jsonl",
               "geometry3k_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules.jsonl",
               "figureqa_en_20240402_extracted_pairs_vqa_correctness_rules.jsonl",
               "infographics_20240403_qa_20240407_v2_extracted_pairs_vqa_correctness_rules.jsonl",
               "unigeo_calc_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules.jsonl",
               "geomverse_extracted_pairs_vqa_correctness_rules.jsonl",
               "iconqa_train_extracted_pairs_vqa_correctness_rules.jsonl",
               "dvqa_en_20240402_extracted_int_only_pairs_vqa_correctness_rules.jsonl",
               "geos_en_20240402_extracted_pairs_vqa_correctness_rules.jsonl",
               ]


# 读取数据
d = read_json(path)
train_dataset_json = []
max_num_per_dataset = 1000  # 每个数据集最多选取1000个样本

# 根据dataset字段将数据条目分组
dataset_groups = {}
for entry in d:
    dataset_name = entry.get('dataset', '')
    dataset_groups.setdefault(dataset_name, []).append(entry)

# 处理每个数据集
for dataset_file in dataset_list:
    dataset_name = dataset_file.replace(".jsonl", "")
    if dataset_name not in dataset_groups:
        continue

    data_entries = dataset_groups[dataset_name]

    # 分离三类数据
    non_01_entries = [e for e in data_entries if e.get('accuracy', 2) not in (0.0, 1.0)]
    zero_entries = [e for e in data_entries if e.get('accuracy') == 0.0]
    one_entries = [e for e in data_entries if e.get('accuracy') == 1.0]

    # 第一阶段：优先选择非0/1样本
    random.shuffle(non_01_entries)
    selected_non_01 = non_01_entries[:max_num_per_dataset]
    selected_entries = selected_non_01.copy()
    remaining = max_num_per_dataset - len(selected_entries)

    # 第二阶段：补充等量0/1样本
    if remaining > 0:
        random.shuffle(zero_entries)
        random.shuffle(one_entries)

        # 计算最大可补充的对数
        possible_pairs = min(len(zero_entries), len(one_entries), remaining // 2)
        selected_zero = zero_entries[:possible_pairs]
        selected_one = one_entries[:possible_pairs]
        selected_entries.extend(selected_zero + selected_one)
        remaining -= possible_pairs * 2

        # 第三阶段：补充剩余数量（允许不等量）
        if remaining > 0:
            combined_remaining = (
                    zero_entries[possible_pairs:] +
                    one_entries[possible_pairs:]
            )
            random.shuffle(combined_remaining)
            selected_entries.extend(combined_remaining[:remaining])

    # 最终保证不超过最大数量
    selected_entries = selected_entries[:max_num_per_dataset]
    print(len(selected_entries))
    train_dataset_json.extend(selected_entries)

# 写入处理后的数据
write_json("reweighting/MMPR/InternVL-MPO/train.json", train_dataset_json)
