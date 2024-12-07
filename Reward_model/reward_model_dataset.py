import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image


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
        add = self.data_js[idx]['add']
        image_path = self.data_js[idx]['image_path']

        if os.path.isfile(image_path):
            image = Image.open(image_path)
            tag = f"<|image_1|>\n"
            messages = [
                {"role": "user", "content": tag + prompt + "\n\n" + add},
            ]
        else:
            image = None
            messages = [
                {"role": "user", "content": prompt + "\n\n" + add},
            ]

        prompt_answer = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(prompt_answer, image, return_tensors="pt")
        label = self.data_js[idx]["accuracy"]

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'image_sizes': inputs['image_sizes'].squeeze(),
            'label': label
        }