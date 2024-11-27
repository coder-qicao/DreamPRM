import os
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AdamW
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator, DeepSpeedPlugin
import pandas as pd
from PIL import Image
from peft import LoraConfig, get_peft_model
import wandb
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

max_length = 900
dataset = "ScienceQA"
model_id = "models/phi3_5/VM_base_best_checkpoint"

processor = AutoProcessor.from_pretrained(
    "microsoft/Phi-3.5-vision-instruct",
    trust_remote_code=True,
)


# Custom Dataset class
class MyDataset(Dataset):
    def __init__(self, data_js, processor, part):
        self.data_js = data_js
        self.processor = processor
        self.part = part

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        # find question and prompt
        i = self.data_js[idx]['id']
        prompt = self.data_js[idx]['input']
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt_answer = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        if os.path.isfile(f"datasets/{dataset}/images/{self.part}/{i}.png"):
            image = Image.open(f"datasets/{dataset}/images/{self.part}/{i}.png")
        else:
            image = None
        inputs = self.processor(prompt_answer, image, return_tensors="pt")

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'image_sizes': inputs['image_sizes'].squeeze()
        }


class Phi3_5_RM(nn.Module):
    def __init__(self, model_id):
        super(Phi3_5_RM, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="bfloat16",
            _attn_implementation='flash_attention_2'
        )
        self.LN = nn.Linear(self.base_model.config.vocab_size, 1)

    def forward(self, input_ids, attention_mask, pixel_values, image_sizes):
        outputs = self.base_model(input_ids=input_ids,attention_mask=attention_mask, pixel_values = pixel_values, image_sizes = image_sizes).logits[:, -1, :]
        # print(outputs)
        value_outputs = self.LN(outputs)
        # print(value_outputs)
        return value_outputs.squeeze(dim=1)


# Load test set data
test_js = f'results/{dataset}/phi3_5/test_step.json'


def read_json(source):
    with open(source, 'r', encoding='utf-8') as f:
        json_list = json.load(f)
    return json_list


test_json = read_json(test_js)

# Create a custom dataset
test_dataset = MyDataset(test_json, processor,'test')

# Create data loaders
batch_size = 1  # Set batch size
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Set device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model = Phi3_5_RM(model_id)
best_model.LN.load_state_dict(torch.load("models/phi3_5/VM_LN_best_checkpoint.pth"))
best_model.to(device)


# Perform inference
test_results = []
with torch.no_grad():
    for index, batch in enumerate(tqdm(test_dataloader)):
        d = test_json[index]
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        image_sizes = batch['image_sizes'].to(device)
        ture_false = d['true_false']
        outputs = best_model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
                     image_sizes=image_sizes)
        j = {'id':d['id'], 'Step_id':d['Step_id'], 'reward':outputs, 'true_false':ture_false}
        test_results.append(j)
        print(f"Inference results: {j}")

with open(f'results/{dataset}/phi3_5/test_result.json', 'w') as f:
    json.dump(test_results, f)

