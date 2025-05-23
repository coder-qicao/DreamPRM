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
from utils.internVL_utils.load_pretrained_model_and_processor import load_pretrained_model, load_pretrained_tokenizer
from utils.response_collector import ResponseCollector
from reweighting.utils import load_QwenVL_RM, generate_reward_model_input
from utils.split_step import split_step
from transformers import AutoProcessor
import numpy as np

dataset = 'WeMath'
part = 'test'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_json_file_path = f"dataset/{dataset}/{part}.json"
dataset_json = read_json(dataset_json_file_path)
model = load_pretrained_model()
tokenizer = load_pretrained_tokenizer()
responses = ResponseCollector(out_path=F"BoN/results/{dataset}_InternVL_MPO.json")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
reward_model = load_QwenVL_RM(model_id = "reweighting/weights/base_model",
                      LN_id = "reweighting/weights/LN_weights.pt")
reward_model.eval()
model_name = "InternVL-MPO"
N_json = [f"inference/results/{dataset}/{model_name}/0.json", f"inference/results/{dataset}/{model_name}/1.json",
          f"inference/results/{dataset}/{model_name}/2.json", f"inference/results/{dataset}/{model_name}/3.json",
          f"inference/results/{dataset}/{model_name}/4.json", f"inference/results/{dataset}/{model_name}/5.json",
          f"inference/results/{dataset}/{model_name}/6.json", f"inference/results/{dataset}/{model_name}/7.json",]
N = []
gpt_eval_json = {}

for response in N_json:
    f = read_json(response)
    N.append(f)
true_num = 0
sequence_class = []
total_num = 0

for data in dataset_json:
    # example data:
    # {
    #     "id": 0,
    #     "input": "Question: Which figure of speech is used in this text?\nSing, O goddess, the anger of Achilles son
    #     of Peleus, that brought countless ills upon the Achaeans.\n\u2014Homer, The Iliad\nContext: N/A
    #     \nOptions: (A) chiasmus (B) apostrophe\n",
    #     "ground_truth": "B",
    #     "response": "Because Figures of speech are words or phrases that use language in a nonliteral or unusual way.
    #     They can make writing more expressive.\\nAnaphora is the repetition of the same word or words at the beginning
    #     of several phrases or clauses.\\nWe are united. We are powerful. We are winners.\\nAntithesis involves
    #     contrasting opposing ideas within a parallel grammatical structure.\\nI want to help, not to hurt.
    #     \\nApostrophe is a direct address to an absent person or a nonhuman entity. \\nOh, little bird, what makes you
    #     sing so beautifully?\\nAssonance is the repetition of a vowel sound in a series of nearby words.
    #     \\nTry to light the fire.\\nChiasmus is an expression in which the second half parallels the first but
    #     reverses the order of words.\\nNever let a fool kiss you or a kiss fool you.\\nUnderstatement involves
    #     deliberately representing something as less serious or important than it really is.\\nAs you know, it can get
    #     a little cold in the Antarctic. The text uses apostrophe, a direct address to an absent person or a nonhuman
    #     entity.\\nO goddess is a direct address to a goddess, a nonhuman entity."
    # }
    flag = True
    best_answer = ''
    input = data['input']
    image_path = f"dataset/{dataset}/images/{part}/{data['id']}.png"
    if not os.path.isfile(image_path):
        continue
    max_score = -10
    true_N = 0
    for i in N:
        index = 1
        min_score = 10
        response = i[total_num]['response']
        mean_score = 0
        scores = []
        while flag:
            response_step = split_step(index, response)
            if response_step == "":
                if index == 1:
                    min_score = 0.5
                    scores.append(0.5)
                else:
                    mean_score = mean_score / (index - 1)
                break
            else:
                reward_inputs = generate_reward_model_input(input, response_step, image_path, processor)
                input_ids = reward_inputs['input_ids'].to(device)
                attention_mask = reward_inputs['attention_mask'].to(device)
                pixel_values = reward_inputs['pixel_values'].unsqueeze(0).to(device)
                image_grid_thw = reward_inputs['image_grid_thw'].to(device)
                with torch.no_grad():
                    score = reward_model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
                                        image_grid_thw=image_grid_thw)
                    mean_score += np.log(score / (1 - score))
                    scores.append(round(float(score),3))
            index += 1
        if i[total_num]['true_false']:
            true_N += 1
        print(scores, i[total_num]['true_false'])
        sequence_class.append({"sequence":scores, "mean": np.mean(scores), "label": i[total_num]['true_false']})
        if mean_score >= max_score:
            best_answer = response
            true_false = i[total_num]['true_false']
            max_score = mean_score

    total_num += 1
    # true_false = verify_answer(best_answer, data['ground_truth'], data['type']) # for MMVet
    if true_false:
        true_num += 1
    print(f"{data['id']}:{true_false}, True N:{true_N}, total:{true_num}")
    responses.add_response(best_answer, data['input'], data['id'], true_false)
    # gpt_eval_json[data['vid']] = best_answer.split("Final answer: ")[-1]  # for MMVet
    if data['id'] % 100 == 0:
        write_json(responses.get_out_path(), responses.get_responses())

print(f"Accuracy {true_num/(total_num+1) * 100:.2f}%")
write_json(responses.get_out_path(), responses.get_responses())
write_json(f"BoN/results/{dataset}_internVL_sequence.json",sequence_class)
# write_json(f"BoN/results/{dataset}_internVL_gpt_eval.json", gpt_eval_json)  # for MMVet
