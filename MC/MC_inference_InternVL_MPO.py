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
import torch
from utils.json_processor import read_json, write_json
from utils.internVL_utils.load_pretrained_model_and_processor import load_pretrained_model_MPO, load_pretrained_tokenizer_MPO
from utils.internVL_utils.one_shot_CoT_prompt_building import one_shot_prompt_building_single_image_completion
from utils.internVL_utils.generate_response import generate_response
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.verify_answer_by_LLM import judge_vqa_answer

dataset = 'MMPR'
part = 'train'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_json_file_path = f"MC/results/MMPR/Tree/0.json"
dataset_json = read_json(dataset_json_file_path)
image_part = 'train_more_dataset'
image_dataset_json_file_path = f"dataset/{dataset}/{image_part}.json"
image_dataset_json = read_json(image_dataset_json_file_path)
model = load_pretrained_model_MPO()
tokenizer = load_pretrained_tokenizer_MPO()
judge_name = "Qwen/Qwen2.5-7B-Instruct"
judge = AutoModelForCausalLM.from_pretrained(
        judge_name,
        torch_dtype="auto",
        device_map="auto"
    )
judge_tokenizer = AutoTokenizer.from_pretrained(judge_name)

for data in dataset_json:
    # expanding
    if (data['accuracy'] != 0 and data['accuracy'] != 1) or data['times'] < 4:
        input = data['input']
        id = data['id']
        image_path = data['image_path']
        ground_truth = data['ground_truth']
        image_dataset = data['dataset']
        add = data['add']
        if not os.path.isfile(image_path):
            continue
        flag = True
        sid = data['sid']
        prompt = one_shot_prompt_building_single_image_completion(input, add="\n\n" + add)
        response = generate_response(tokenizer, model, prompt, image_path, do_sample=True, temperature=1.0)
        if len(response) > 30 or sid == 5:
            # print(response)
            # print(ground_truth)

            # judge
            candidate_answer = response.split("Final answer: ")[-1]
            result = judge_vqa_answer(input, ground_truth, candidate_answer, judge, judge_tokenizer)
            # print(result)
            if result == "correct":
                true_false = True
            elif result == "incorrect":
                if candidate_answer == ground_truth:
                    true_false = True
                else:
                    true_false = False
            print(f"{id}-Step {sid}:{true_false}")
            # update
            if true_false:
                data['score'] += 1
            data['times'] += 1
            data['accuracy'] = data['score'] / data['times']
            print(data['accuracy'])
    if (data['id'] + 1) % 100 == 0:
        write_json(dataset_json_file_path, dataset_json)

write_json(dataset_json_file_path, dataset_json)
