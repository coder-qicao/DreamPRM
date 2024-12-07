import os
import torch
from utils.json_processor import read_json, write_json
from utils.phi_vision_utils.load_pretrained_model_and_processor import load_pretrained_model, load_pretrained_processor
from Reward_model.reward_model_utils import load_Phi_Vision_RM, generate_reward_model_input
from utils.response_collector import ResponseCollector
from utils.phi_vision_utils.two_shots_prompt_building import two_shots_prompt_building_single_image
from utils.phi_vision_utils.generate_response import generate_response
from utils.split_step import split_step
from utils.verify_answer import verify_answer_multi_choice

os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
path = "/home/qi/python_project/multimodal_reasoning"
os.chdir(path)

dataset = 'ScienceQA'
part = 'test'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_json_file_path = f"dataset/{dataset}/{part}.json"
dataset_json = read_json(dataset_json_file_path)
model = load_pretrained_model()
processor = load_pretrained_processor()
reward_model = load_Phi_Vision_RM()
reward_model.eval()
responses = ResponseCollector(out_path="BoN/results/ScienceQA.json")

for data in dataset_json:
    # example data:
    # {
    #     "id": 0,
    #     "input": "Question: Which figure of speech is used in this text?\nSing, O goddess, the anger of Achilles son of Peleus, that brought countless ills upon the Achaeans.\n\u2014Homer, The Iliad\nContext: N/A\nOptions: (A) chiasmus (B) apostrophe\n",
    #     "ground_truth": "B",
    #     # "reasoning": "Because Figures of speech are words or phrases that use language in a nonliteral or unusual way. They can make writing more expressive.\\nAnaphora is the repetition of the same word or words at the beginning of several phrases or clauses.\\nWe are united. We are powerful. We are winners.\\nAntithesis involves contrasting opposing ideas within a parallel grammatical structure.\\nI want to help, not to hurt.\\nApostrophe is a direct address to an absent person or a nonhuman entity.\\nOh, little bird, what makes you sing so beautifully?\\nAssonance is the repetition of a vowel sound in a series of nearby words.\\nTry to light the fire.\\nChiasmus is an expression in which the second half parallels the first but reverses the order of words.\\nNever let a fool kiss you or a kiss fool you.\\nUnderstatement involves deliberately representing something as less serious or important than it really is.\\nAs you know, it can get a little cold in the Antarctic. The text uses apostrophe, a direct address to an absent person or a nonhuman entity.\\nO goddess is a direct address to a goddess, a nonhuman entity."
    #     # "hint", "question type"
    # }
    flag = True
    index = 1
    N = 4
    best_answer = ''
    input = data['input']
    image_path = f"dataset/{dataset}/images/{part}/{data['id']}.png"
    if not os.path.isfile(image_path):
        continue
    while flag:
        prompt, image = two_shots_prompt_building_single_image(input, image_path, processor, add=best_answer)
        max_score = -10
        for i in range(N):
            response = generate_response(processor, model, prompt, image, temperature=0.2, do_sample=True)
            response_step = split_step(index, response)
            step_prompt, image = two_shots_prompt_building_single_image(input, image_path, processor, add=best_answer+response_step)
            if response_step == "":
                flag = False
                reward_inputs = generate_reward_model_input(step_prompt, image, processor)
                input_ids = reward_inputs['input_ids'].unsqueeze(0).to(device)
                attention_mask = reward_inputs['attention_mask'].unsqueeze(0).to(device)
                pixel_values = reward_inputs['pixel_values'].unsqueeze(0).to(device)
                image_sizes = reward_inputs['image_sizes'].unsqueeze(0).to(device)
                with torch.no_grad():
                    score = reward_model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
                                         image_sizes=image_sizes)
                if score > max_score:
                    max_score = score
                    candidate = response
            else:
                flag = True
                reward_inputs = generate_reward_model_input(step_prompt, image, processor)
                input_ids = reward_inputs['input_ids'].unsqueeze(0).to(device)
                attention_mask = reward_inputs['attention_mask'].unsqueeze(0).to(device)
                pixel_values = reward_inputs['pixel_values'].unsqueeze(0).to(device)
                image_sizes = reward_inputs['image_sizes'].unsqueeze(0).to(device)
                with torch.no_grad():
                    score = reward_model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
                                        image_sizes=image_sizes)
                if score > max_score:
                    max_score = score
                    candidate = response_step
        index += 1
        best_answer += candidate

    true_false = verify_answer_multi_choice(best_answer, data['ground_truth'])
    print(f"{data['id']}:{true_false}")
    responses.add_response(best_answer, data['input'], data['id'], true_false)
    if data['id'] % 100 == 0:
        write_json(responses.get_out_path(), responses.get_responses())

write_json(responses.get_out_path(), responses.get_responses())
