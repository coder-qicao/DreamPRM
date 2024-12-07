import os
import torch
from utils.json_processor import read_json, write_json
from utils.phi_vision_utils.load_pretrained_model_and_processor import load_pretrained_model, load_pretrained_processor
from Reward_model.reward_model_utils import load_Phi_Vision_RM, generate_reward_model_input
from utils.response_collector import ResponseCollector
from utils.phi_vision_utils.two_shots_prompt_building import two_shots_prompt_building_single_image
from utils.phi_vision_utils.generate_response import generate_response
from utils.split_step import split_step
from utils.verify_answer import verify_answer_multi_choice, verify_answer

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
path = "/home/qi/python_project/multimodal_reasoning"
os.chdir(path)

dataset = 'MathVista'
part = 'testmini'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_json_file_path = f"dataset/{dataset}/{part}.json"
dataset_json = read_json(dataset_json_file_path)
model = load_pretrained_model()
processor = load_pretrained_processor()
responses = ResponseCollector(out_path="results/MathVista/Phi_vision.json")

for data in dataset_json:
    # example data:
    # {
    #     "id": 0,
    #     "input": "Question: Which figure of speech is used in this text?\nSing, O goddess, the anger of Achilles son of Peleus, that brought countless ills upon the Achaeans.\n\u2014Homer, The Iliad\nContext: N/A\nOptions: (A) chiasmus (B) apostrophe\n",
    #     "ground_truth": "B",
    #     # "reasoning": "Because Figures of speech are words or phrases that use language in a nonliteral or unusual way. They can make writing more expressive.\\nAnaphora is the repetition of the same word or words at the beginning of several phrases or clauses.\\nWe are united. We are powerful. We are winners.\\nAntithesis involves contrasting opposing ideas within a parallel grammatical structure.\\nI want to help, not to hurt.\\nApostrophe is a direct address to an absent person or a nonhuman entity.\\nOh, little bird, what makes you sing so beautifully?\\nAssonance is the repetition of a vowel sound in a series of nearby words.\\nTry to light the fire.\\nChiasmus is an expression in which the second half parallels the first but reverses the order of words.\\nNever let a fool kiss you or a kiss fool you.\\nUnderstatement involves deliberately representing something as less serious or important than it really is.\\nAs you know, it can get a little cold in the Antarctic. The text uses apostrophe, a direct address to an absent person or a nonhuman entity.\\nO goddess is a direct address to a goddess, a nonhuman entity."
    #     # "hint", "type"
    # }
    input = data['input']
    image_path = f"dataset/{dataset}/images/{part}/{data['id']}.png"
    if not os.path.isfile(image_path):
        continue
    prompt, image = two_shots_prompt_building_single_image(input, image_path, processor, type=data['type'], hint=data['hint'])
    response = generate_response(processor, model, prompt, image, temperature=0.0, do_sample=False)
    print(response)
    true_false = verify_answer(response, data['ground_truth'], data['type'])
    print(f"{data['id']}:{true_false}")
    responses.add_response(response, data['input'], data['id'], true_false)
    if data['id'] % 100 == 0:
        write_json(responses.get_out_path(), responses.get_responses())

write_json(responses.get_out_path(), responses.get_responses())
