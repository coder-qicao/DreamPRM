'''
python3 data.py --input_json data/openmathinstruct-1.json --out_json data/prefixes.json --openmath
'''

import copy
import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from qwen_vl_utils import process_vision_info
import json
from transformers import AutoProcessor
from PIL import Image
import re

def read_json(source):
    with open(source, 'r', encoding='utf-8') as f:
        return json.load(f)

def split_step(s_id, response):
    s = f"Step {s_id}"
    s_next = f"Step {s_id+1}"
    if s_next in response:
        assistant = response.split(s_next)[0]
    elif "Final answer" in response and s in response:
        assistant = response.split("Final answer")[0]
    else:
        assistant = ""
    return assistant

def find_max_step(response):
    """
    Find the maximum step number in a response string containing steps.

    Args:
        response: String containing steps in formats like "Step 1: ...", "Step 2: ...", etc.

    Returns:
        Integer representing the highest step number found. Returns 0 if no steps are found.
    """
    # Find all occurrences of step patterns (case-insensitive)
    # Matches: "Step 1", "STEP 2", "step3", "Step: 4", etc.
    step_numbers = re.findall(r'Step[\s:]*(\d+)', response, re.IGNORECASE)
    # Return 0 if no step numbers found
    if not step_numbers:
        return 0
    # Convert found numbers from strings to integers
    step_numbers = [int(num) for num in step_numbers]
    # Return the maximum step number
    return max(step_numbers)

def split_steps(response):
    max_step = find_max_step(response)
    return [split_step(i, response) for i in range(1, max_step+1) if split_step(i, response)]

def split_steps_openmath(sol):
    parts = re.split(r'<llm-code>', sol)[1:]
    steps = []
    for part in parts:
        code, rest = part.split('</llm-code>', 1)
        steps.append(code.strip())
        if '<llm-code-output>' in rest:
            out = rest.split('<llm-code-output>',1)[1].split('</llm-code-output>',1)[0]
            steps.append(out.strip())
    return steps

def build_prefix_dataset(src_json, out_json, openmath=False):
    data = read_json(src_json)
    prefixes = []
    for idx, itm in enumerate(data):
        sol = itm.get('generated_solution', itm.get('solution', ''))
        gt  = str(itm.get('expected_answer', itm.get('answer', '')))
        steps = split_steps_openmath(sol) if openmath else split_steps(sol)
        for k in range(len(steps)):
            p = "\n".join(steps[:k+1])
            prefixes.append({'qid':str(idx), 'prefix':p, 'ground_truth':gt})
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(prefixes, f, ensure_ascii=False, indent=2)

def resize_image_if_needed(img, max_size=512):
    """
    Resize an image proportionally if either width or height exceeds max_size.
    Maintains the original aspect ratio while scaling down the longest side to max_size.

    :param img: PIL.Image object to be resized
    :param max_size: Maximum allowed length for the longest side (default: 512)
    :return: Resized PIL.Image object
    """
    width, height = img.size
    # Check if the longest dimension exceeds max_size
    if max(width, height) > max_size:
        # Calculate scaling ratio while maintaining aspect ratio
        scale_ratio = max_size / float(max(width, height))
        new_width = int(width * scale_ratio)
        new_height = int(height * scale_ratio)
        # Resize image using LANCZOS resampling for high quality
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img

class MyDataset_QwenVL(Dataset):
    def __init__(self, data_js, processor):
        self.data_js = data_js
        self.processor = processor

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        # find question and prompt
        i = self.data_js[idx]['id']
        prompt = self.data_js[idx]['input']
        add = self.data_js[idx]['add']
        image_path = self.data_js[idx]['image_path']
        prompt = prompt + "\n\n" + add
        label = self.data_js[idx]['accuracy']
        dset = self.data_js[idx]['dataset']

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        image_inputs = [resize_image_if_needed(image_inputs[0])]
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'image_grid_thw': inputs['image_grid_thw'].squeeze(),
            'label': label,
            'dataset': dset
        }

class MyDataset_Llava(Dataset):
    def __init__(self, data_js, processor):
        self.data_js = data_js
        self.processor = processor

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        # find question and prompt
        i = self.data_js[idx]['id']
        prompt = self.data_js[idx]['input']
        add = self.data_js[idx]['add']
        image_path = self.data_js[idx]['image_path']
        prompt = prompt + "\n\n" + add
        label = self.data_js[idx]['accuracy']
        dset = self.data_js[idx]['dataset']

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        raw_image = Image.open(image_path)
        inputs = self.processor(images=raw_image, text=text, return_tensors='pt').to(0, torch.bfloat16)

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'image_sizes': inputs['image_sizes'].squeeze(),
            'label': label,
            'dataset': dset
        }

class MyDataset_QwenMath(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512):
        self.data = read_json(json_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        itm = self.data[idx]
        enc = self.tokenizer(itm['prefix'], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'qid':itm['qid'], 
            'input_ids':enc['input_ids'].squeeze(0), 
            'attention_mask':enc['attention_mask'].squeeze(0)
            }


class MyMetaDataset_QwenVL(Dataset):
    def __init__(self, data_js, processor):
        self.data_js = data_js
        self.processor = processor

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        # find question and prompt
        input = self.data_js[idx]['input']
        image_path = self.data_js[idx]['image_path']
        label = self.data_js[idx]['true_false']

        r_dict = {}
        step_num = find_max_step(input)
        for index in range(step_num):
            step = split_step(index+1, input)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": step},
                    ],
                }
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            image_inputs = [resize_image_if_needed(image_inputs[0])]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            r_dict[f"{index+1}"] = {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'pixel_values': inputs['pixel_values'].squeeze(),
                'image_grid_thw': inputs['image_grid_thw'].squeeze(),
            }
        r_dict["labels"] = torch.tensor(label).to(dtype=torch.float)
        return r_dict

class MyMetaDataset_Llava(Dataset):
    def __init__(self, data_js, processor, step_num = 5):
        self.data_js = data_js
        self.processor = processor
        self.step_num = step_num

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        # find question and prompt
        input = self.data_js[idx]['input']
        image_path = self.data_js[idx]['image_path']
        label = self.data_js[idx]['true_false']
        r_dict = {}
        for index in range(self.step_num):
            step = split_step(index+1, input)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": step},
                        {"type": "image"},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            raw_image = Image.open(image_path)
            inputs = self.processor(images=raw_image, text=text, return_tensors='pt').to(0, torch.bfloat16)
            r_dict[f"{index+1}"] = {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'pixel_values': inputs['pixel_values'].squeeze(),
                'image_sizes': inputs['image_sizes'].squeeze(),
            }
        r_dict["labels"] = torch.tensor(label).to(dtype=torch.float)
        return r_dict

class MyMetaDataset_QwenMath(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512):
        self.meta = read_json(json_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        itm = self.meta[idx]
        enc = self.tokenizer(itm['prefix'], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'qid':itm['qid'], 
            'input_ids':enc['input_ids'].squeeze(0), 
            'attention_mask':enc['attention_mask'].squeeze(0), 
            'labels':torch.tensor(float(itm['reward']),dtype=torch.float)
            }

def build_dataloader(
        processor_path,
        train_json_file,
        meta_json_file,
        train_batch_size,
        meta_batch_size,
        openmath=False,
):
    processor = AutoProcessor.from_pretrained(processor_path)
    if openmath:
        build_prefix_dataset(train_json_file, 'prefixes.json', openmath=True)
        train_dataset = read_json('prefixes.json')
        train_dataloader = DataLoader(MyDataset_QwenMath('prefixes.json', processor.tokenizer), batch_size=train_batch_size, shuffle=True)
        meta_dataloader = DataLoader(MyMetaDataset_QwenMath(meta_json_file, processor.tokenizer), batch_size=meta_batch_size, shuffle=True)
    else:
        train_dataset = MyDataset_QwenVL(read_json(train_json_file), processor)
        meta_dataset = MyMetaDataset_QwenVL(read_json(meta_json_file), processor)
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        meta_dataloader = DataLoader(meta_dataset, batch_size=meta_batch_size, shuffle=True)

    return train_dataloader, meta_dataloader
