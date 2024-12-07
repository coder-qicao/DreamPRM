import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
path = "/home/q9cao/python_project/multimodal_reasoning"
os.chdir(path)

import torch
from utils.json_processor import read_json, write_json
from utils.internVL_utils.load_pretrained_model_and_processor import load_pretrained_model, load_pretrained_tokenizer
from utils.response_collector import ResponseCollector
from utils.internVL_utils.one_shot_CoT_prompt_building import one_shot_prompt_building_single_image
from utils.internVL_utils.generate_response import generate_response
from utils.split_step import split_step
from utils.verify_answer import verify_answer_multi_choice, verify_answer


dataset = 'MMPR'
part = 'validation_dataset'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_json_file_path = f"dataset/{dataset}/{part}.json"
dataset_json = read_json(dataset_json_file_path)
model = load_pretrained_model()
tokenizer = load_pretrained_tokenizer()
responses = ResponseCollector(out_path="inference/results/MMPR/InternVL-MPO/CoT.json")

for data in dataset_json:
    input = data['input']
    image_path = data['image_path']
    if not os.path.isfile(image_path):
        continue
    prompt= one_shot_prompt_building_single_image(input)
    response = generate_response(tokenizer, model, prompt, image_path, do_sample=False, temperature=0)
    print(response)
    true_false = verify_answer(response, data['ground_truth'], question_type='MMPR')
    print(f"{data['id']}:{true_false}")
    responses.add_response(response, data['input'], data['id'], true_false)
    if data['id'] % 100 == 0:
        write_json(responses.get_out_path(), responses.get_responses())

write_json(responses.get_out_path(), responses.get_responses())
