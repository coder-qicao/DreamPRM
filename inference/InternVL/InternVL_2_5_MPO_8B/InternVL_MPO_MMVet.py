import sys
sys.path.append("/home/q9cao/python_project/multimodal_reasoning")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
path = "/home/q9cao/python_project/multimodal_reasoning"
os.chdir(path)

import torch
from utils.json_processor import read_json, write_json
from utils.internVL_utils.load_pretrained_model_and_processor import load_pretrained_model_MPO, load_pretrained_tokenizer_MPO
from utils.response_collector import ResponseCollector
from utils.internVL_utils.one_shot_CoT_prompt_building import one_shot_prompt_building_single_image
from utils.internVL_utils.generate_response import generate_response
from utils.verify_answer import verify_answer

dataset = 'MMVet'
part = 'test'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_json_file_path = f"dataset/{dataset}/{part}.json"
dataset_json = read_json(dataset_json_file_path)
model = load_pretrained_model_MPO()
tokenizer = load_pretrained_tokenizer_MPO()

for i in range(8):
    responses = ResponseCollector(out_path=f"inference/results/MMVet/InternVL-MPO/{i}.json")
    gpt_eval_json = {}
    for data in dataset_json:
        input = data['input']
        image_path = f"dataset/{dataset}/images/{part}/{data['id']}.png"
        if not os.path.isfile(image_path):
            continue
        prompt= one_shot_prompt_building_single_image(input)
        response = generate_response(tokenizer, model, prompt, image_path, do_sample=True, temperature=0.3)
        print(response)
        true_false = verify_answer(response, data['ground_truth'], None)
        print(f"{data['id']}:{true_false}")
        responses.add_response(response, data['input'], data['id'], true_false)
        if data['id'] % 100 == 0:
            write_json(responses.get_out_path(), responses.get_responses())
        gpt_eval_answer = response.split("Final answer: ")[-1]
        gpt_eval_id = data['vid']
        gpt_eval_json[gpt_eval_id] = gpt_eval_answer
    write_json(responses.get_out_path(), responses.get_responses())
    write_json(f"inference/results/MMVet/InternVL-MPO/{i}_gpt.json", gpt_eval_json)
