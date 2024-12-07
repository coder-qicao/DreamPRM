import torch
from transformers import AutoProcessor, AdamW
from reweighting.data import MyDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from reweighting.model import QwenVL_RM
import os
import json
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
path = "/home/qi/python_project/multimodal_reasoning"
os.chdir(path)


def read_json(source):
    with open(source, 'r', encoding='utf-8') as f:
        json_list = json.load(f)
    return json_list

path = "Qwen/Qwen2-VL-2B-Instruct"
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
processor = AutoProcessor.from_pretrained(path)
# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

train_dataset = MyDataset(read_json("reweighting/MMPR/train.json"), processor, "train")
batch_size = 1  # Set batch size
learning_rate = 5e-7
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
VM = QwenVL_RM(device)
VM.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = AdamW(VM.parameters(), lr=learning_rate)

for epoch in range(10):
    # Training
    VM.train()
    train_loss = 0.0
    for batch in tqdm(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        image_grid_thw = batch['image_grid_thw'].to(device)
        labels = batch['label'].to(dtype=torch.float).to(device)
        optimizer.zero_grad()
        outputs = VM(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
