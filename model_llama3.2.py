import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import json
import os
import wandb
import re


class llama3_2_11B:
    def __init__(self):
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto", )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.responses = []

    def prompt_building(self, data, dataset):
        i = data
        e = ("Question: How long does it take to make a paper airplane?\nContext: Select the better estimate.\nChoices:\n(A) 45 seconds\n(B) 45 hours\nSolution: The better estimate for how long it takes to make a paper airplane is 45 seconds. 45 hours is too slow.\nAnswer:(A)\n"
             "Question: What is the largest city in the nation where this plane is headquartered?\nContext: N/A\nChoices:\n(A) hong kong\n(B) osaka\n(C) shanghai\n(D) tokyo\nSolution: The text on the image says \"Japan. Endless Discovery\". This indicates that the plane is headquartered in Japan. \n\nAmong the Japanese cities, Tokyo is the largest city.\n\nThus, the answer is D (tokyo).\nAnswer:(D)\n")
        if os.path.isfile(f"{dataset}/images/{i['id']}.png"):
            image = Image.open(f"{dataset}/images/{i['id']}.png")
            prompt = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{e}{i['input']}Hint: Please answer the question and provide the correct option letter, e.g., (A), (B), (C), (D), at the end."}
                ]}
            ]
        else:
            image = None
            prompt = [
                {"role": "user", "content": [
                    {"type": "text", "text": f"{e}{i['input']}Hint: Please answer the question and provide the correct option letter, e.g., (A), (B), (C), (D), at the end."}
                ]}
            ]
        return prompt, image

    def generate_response(self, prompt, image, i):
        input_text = self.processor.apply_chat_template(prompt, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to('cuda:0')

        output = self.model.generate(**inputs, max_new_tokens=1000)
        answer = self.processor.decode(output[0])
        self.responses.append({"id":i, "response": answer})
        return answer

    def validate_answer(self, answer, i):
        marker = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        # 找到 marker 最后出现的位置
        start_pos = answer.rfind(marker)

        if start_pos != -1:  # 确保找到了 marker
            substring = answer[start_pos + len(marker):]
            print(substring)

            # 找出所有非"Answer"前缀的ABCD
            matches = re.findall(r'(?<!Answer)([ABCD])', substring)
            answer = matches[-1] if matches else None  # 获取最后一个匹配项

            if answer == i['ground_truth']:
                print(f'{i['id']}:Bingo!')
                self.responses[i['id']-9701]["true_false"] = True
                right_id.append(i['id'])
            else:
                print(f'{i['id']}:Not Bingo!')
                self.responses[i['id']-9701]["true_false"] = False
                wrong_id.append(i['id'])

            # 记录结果
            wandb.log({
                "right": len(right_id),
                "wrong": len(wrong_id),
                "accuracy": len(right_id) / (len(right_id) + len(wrong_id)) * 100
            })
        else:
            print("Marker not found.")

    def save_response_as_json(self, path, part):
        with open(f"{path}{part}.json", "w") as f:
            json.dump(self.responses, f, indent=4)


if __name__ == '__main__':
    dataset = "datasets/ScienceQA"
    with open(f"{dataset}/train.json", "r") as f:
        data = json.load(f)
    model = llama3_2_11B()

    right_id = []
    wrong_id = []
    wandb.init(
        project="multimodal_reasoning_zero_shot_llama3.2"
    )

    for i in data[9701:]:
        prompt, image = model.prompt_building(i, dataset)
        answer = model.generate_response(prompt, image, i['id'])
        model.validate_answer(answer, i)
        if i['id'] % 100 == 0:
            model.save_response_as_json(f"results/ScienceQA","/validate")

    print(f"We got {len(right_id)} right answers, {len(wrong_id)} wrong answers.")
    print(f"The accuracy is {len(right_id) / (len(right_id) + len(wrong_id)) * 100:.2f}%")

