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
from utils.verify_answer_by_LLM import judge_vqa_answer
from transformers import AutoTokenizer, AutoModelForCausalLM

dataset = "MathVision"
model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset_path = f"dataset/{dataset}/test.json"
d = read_json(dataset_path)
for j in range(1):
    file_path = f"inference/results/{dataset}/InternVL-MPO/{j}.json"
    data = read_json(file_path)

    # original accuracy
    true_num = 0
    false_num = 0

    for i in data:
        flag = i['id']
        question = d[flag]["input"]
        ground_truth = d[flag]["ground_truth"]
        candidate_answer = i["response"].split("Final answer: ")[-1]
        result = judge_vqa_answer(question, ground_truth, candidate_answer, model, tokenizer)
        # print(question)
        print(f"{candidate_answer} vs {ground_truth}: {result}")
        if result == "correct":
            if candidate_answer != ground_truth:
                print()
                # print(question)
            i['true_false'] = True
            true_num += 1
        elif result == "incorrect":
            if candidate_answer == ground_truth:
                true_num += 1
                i['true_false'] = True
            else:
                i['true_false'] = False
                false_num += 1
    print(f"Accuracy: {true_num / (true_num + false_num)}")
    write_json(file_path, data)