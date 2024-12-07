import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import json
import os
import wandb
import re
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'

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

    def prompt_building_step(self, data, data_step, dataset, part):
        i = data
        placeholder = f"<|image_1|>\n"
        e = (
            "Question: What do these two changes have in common?\ntearing a piece of paper\nbreaking a piece of glass\nContext: N/A\nOptions: (A) Both are only physical changes. (B) Both are chemical changes. (C) Both are caused by cooling. (D) Both are caused by heating.\nStep 1: Identify the trait mentioned in the description.\nThe trait mentioned is \"Logan has blue eyes.\"\n\nStep 2: Determine which option provides information about the origin of the trait.\nOption A states that Logan's mother has blue eyes and passed this trait down to Logan.\n\nStep 3: Compare the information in the options to the trait mentioned.\nOption A provides information about the origin of the trait, while option B does not.\n\nStep 4: Select the option that supports the conclusion that Logan inherited the trait.\n(A) supports the conclusion that Logan inherited the trait.\n\nAnswer: (A)\n\n"
            "Question: What information supports the conclusion that Logan inherited this trait?\nContext: Read the description of a trait.\nLogan has blue eyes.\nOptions: (A) Logan's mother has blue eyes. She passed this trait down to Logan. (B) Logan likes to wear a blue sweater to match his blue eyes.\nStep 1: Identify the type of change for each option.\nTearing a piece of paper is a physical change because it changes the shape and size of the paper, but it does not change the chemical composition of the paper.\nBreaking a piece of glass is also a physical change because it changes the shape and size of the glass, but it does not change the chemical composition of the glass.\n\nStep 2: Compare the types of changes.\nBoth tearing a piece of paper and breaking a piece of glass are physical changes.\n\nStep 3: Select the correct option.\nThe correct option is (A).\n\nAnswer:(A)\n\n"
        )
        prompts = []
        if os.path.isfile(f"datasets/{dataset}/images/{part}/{i['id']}.png"):
            image = [Image.open(f"datasets/{dataset}/images/{part}/{i['id']}.png")]
            text = i['input']
            response = data_step['response']
            query = e + text + "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end. Let's answer step by step.\n"
            s_id = 1
            s = f"Step {s_id}"
            while s in response:
                s_id += 1
                s = f"Step {s_id}"
                if s in response:
                    assistant = response.split(s)[0]
                elif "Answer" in response:
                    assistant = response.split("Answer")[0]
                else:
                    return None, None
                messages = [
                    {"role": "user", "content": placeholder + query + assistant},
                ]
                prompt = self.processor.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(prompt)
            return prompts, image
        else:
            return None, None



    def generate_response(self, prompt, images, i, j, s_id):
        inputs = self.processor(prompt, images, return_tensors="pt").to("cuda")

        generation_args = {
            "max_new_tokens": 1000,
            "temperature": 0.7,
            "do_sample": True,
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
        # self.responses.append({"id":i,"id_step_1":j,"response": response, "previous":prompt})  # step 0
        self.responses.append({"id": i['id'], "step_id":s_id, "mc_id": j, "response": response, "previous": prompt})
        return response

    def save_response_as_json(self, path, part):
        with open(f"{path}{part}_step_2.json", "w") as f:
            json.dump(self.responses, f, indent=4)

    def validate_answer(self, answer, i, index):
        # 找出所有非"Answer"前缀的ABCD
        matches = re.findall(r'(?<!Answer)([ABCD])', answer)
        answer = matches[-1] if matches else None  # 获取最后一个匹配项

        if answer == i['ground_truth']:
            print(f"{i['id']}:Bingo!")
            self.responses[index]["true_false"] = True
        else:
            print(f"{i['id']}:Not Bingo!")
            self.responses[index]["true_false"] = False
        return self.responses[index]["true_false"]




if __name__ == '__main__':
    dataset = "ScienceQA"
    part = "train"
    step = "step_1"
    with open(f"datasets/{dataset}/{part}.json", "r") as f:
        data_question = json.load(f)
    with open(f"results/{dataset}/phi3_5/{part}_2_good.json", "r") as f:
        data_step = json.load(f)
    model = phi_3_5_vision_instruct()

    wandb.init(
     project="ScienceQA_two_shots_MCTS_phi3.5"
    )
    reward_list = []
    index = 0

    for i in data_step[400:]:
        prompts, image = model.prompt_building_step(data_question[i['id']], i, dataset, part)
        if prompts is None:
            continue
        k = 1 if i["true_false"] else 0
        if image is None:
            continue
        s_id = 1
        for prompt in prompts:
            reward = {'id': i['id']}
            r = k
            for j in range(7):
                response = model.generate_response(prompt, image, i, j+1, s_id)
                print(response)
                if model.validate_answer(response, data_question[i['id']], index):
                    r += 1
                index += 1
            wandb.log({
                "reward": r/8
            })
            reward[f'Step {s_id}'] = r/8
            reward_list.append(reward)
            print(f"\nReward: {r}/8")
            s_id += 1
        if index % 100 == 0:
            model.save_response_as_json(f"results/ScienceQA/phi3_5/", part)
            with open("results/ScienceQA/phi3_5/reward_2.json", "w") as f:
                json.dump(reward_list, f, indent=4)
    model.save_response_as_json(f"results/ScienceQA/phi3_5/", part)
    with open("results/ScienceQA/phi3_5/reward_2.json", "w") as f:
        json.dump(reward_list, f, indent=4)

    # print(f"We got {len(right_id)} right answers, {len(wrong_id)} wrong answers.")
    # print(f"The accuracy is {len(right_id) / (len(right_id) + len(wrong_id)) * 100:.2f}%")

