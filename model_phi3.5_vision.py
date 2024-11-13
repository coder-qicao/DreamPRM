import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import json
import os
import wandb
import re


class phi_3_5_vision_instruct:
    def __init__(self):
        model_id = "microsoft/Phi-3.5-vision-instruct"

        # Note: set _attn_implementation='eager' if you don't have flash_attn installed
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation='eager'
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_crops=16
        )
        self.responses = []

    def prompt_building(self, data, dataset, part):
        i = data
        placeholder = f"<|image_1|>\n"
        if os.path.isfile(f"{dataset}/images{part}/{i['id']}.png"):
            image = [Image.open(f"{dataset}/images{part}/{i['id']}.png")]
            text = i['input']
            query = text + "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
            messages = [
                {"role": "user", "content": placeholder + query},
            ]

            prompt = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt, image

    def prompt_building_perference(self, data, dataset, part):
        i = data
        placeholder = f"<|image_1|>\n"
        if os.path.isfile(f"{dataset}/images{part}/{i['id']}.png"):
            image = [Image.open(f"{dataset}/images{part}/{i['id']}.png")]
            text = i['input']
            query_1 = "Describe the image."
            query_2 = "Describe the image for the following question:" + text
            messages_1 = [
                {"role": "user", "content": placeholder + query_1},
            ]
            messages_2 = [
                {"role": "user", "content": placeholder + query_2},
            ]

            prompt_1 = self.processor.tokenizer.apply_chat_template(
                messages_1,
                tokenize=False,
                add_generation_prompt=True
            )
            prompt_2 = self.processor.tokenizer.apply_chat_template(
                messages_2,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt_1, prompt_2, image

    def generate_response(self, prompt, images, i):
        inputs = self.processor(prompt, images, return_tensors="pt").to("cuda:0")

        generation_args = {
            "max_new_tokens": 1000,
            "temperature": 0.0,
            "do_sample": False,
        }

        generate_ids = self.model.generate(**inputs,
                                      eos_token_id=self.processor.tokenizer.eos_token_id,
                                      **generation_args
                                      )

        # remove input tokens
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids,
                                          skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
        self.responses.append({"id":i,"response": response})
        return response

    def save_response_as_json(self, path, part):
        with open(f"{path}{part}.json", "w") as f:
            json.dump(self.responses, f, indent=4)


    # def validate_response(self, response, i):
    #     answer = response[-1]
    #     if answer == i['ground_truth']:
    #         print('Bingo!')
    #         right_id.append(i['id'])
    #     else:
    #         print('Not Bingo!')
    #         wrong_id.append(i['id'])
    #     wandb.log({
    #         "right": len(right_id),
    #         "wrong": len(wrong_id),
    #         "accuracy": len(right_id) / (len(right_id) + len(wrong_id)) * 100
    #     })


if __name__ == '__main__':
    dataset = "datasets/ScienceQA"
    part = "/train"
    with open(f"{dataset}/train.json", "r") as f:
        data = json.load(f)
    model = phi_3_5_vision_instruct()

    # right_id = []
    # wrong_id = []
    # wandb.init(
    #  project="MathVista_zero_shot_phi3.5"
    # )
    index = 0
    print(index)

    for i in data:
        prompt_1, image = model.prompt_building(i, dataset, part)
        response_1 = model.generate_response(prompt_1, image, index)
        print(response_1)
        index += 1
        print(index)
        if index % 100 == 0:
            model.save_response_as_json(f"results/ScienceQA", part)

    # print(f"We got {len(right_id)} right answers, {len(wrong_id)} wrong answers.")
    # print(f"The accuracy is {len(right_id) / (len(right_id) + len(wrong_id)) * 100:.2f}%")

