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
from utils.response_collector import ResponseCollector
from utils.internVL_utils.one_shot_CoT_prompt_building import one_shot_prompt_building_single_image
from utils.internVL_utils.generate_response import generate_response
from utils.verify_answer import verify_answer

dataset = 'MathVision'
part = 'test'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_json_file_path = f"dataset/{dataset}/{part}.json"
dataset_json = read_json(dataset_json_file_path)
model = load_pretrained_model_MPO()
tokenizer = load_pretrained_tokenizer_MPO()

for i in range(8):
    responses = ResponseCollector(out_path=f"inference/results/MathVision/InternVL-MPO/{i}.json")
    for data in dataset_json:
        input = data['input']
        image_path = f"dataset/{dataset}/images/{part}/{data['id']}.png"
        if not os.path.isfile(image_path):
            continue
        prompt= one_shot_prompt_building_single_image(input)
        response = generate_response(tokenizer, model, prompt, image_path, do_sample=True, temperature=1.0, max_new_tokens=2048)
        print(response)
        true_false = verify_answer(response, data['ground_truth'], None)
        print(f"{data['id']}:{true_false}")
        responses.add_response(response, data['input'], data['id'], true_false)
        if data['id'] % 100 == 0:
            write_json(responses.get_out_path(), responses.get_responses())
    write_json(responses.get_out_path(), responses.get_responses())
