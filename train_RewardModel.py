import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, AutoProcessor, AdamW
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image

max_length = 900
dataset = "ScienceQA"
part = "train"

# Load the pre-trained Llama3.2
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto", )
processor = AutoProcessor.from_pretrained(model_id)

# Custom Dataset class
class MyDataset(Dataset):
    def __init__(self, data_js, processor, part):
        self.data_js = data_js
        self.processor = processor
        self.part = part

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        prompt_answer = self.data_js[idx]['response']
        if os.path.isfile(f"datasets/{dataset}/{self.part}/images/{idx}.png"):
            image = Image.open(f"datasets/{dataset}/{self.part}/images/{idx}.png")
        else:
            image = None

        inputs = self.processor(
            image,
            prompt_answer,
            add_special_tokens=False,
            return_tensors="pt"
        ).to('cuda:0')
        label = self.data_js[idx]['true_false']
        if label:
            label = 1
        else:
            label = -1

        return {
            'inputs': inputs,
            'label': label
        }


class Llama3_2_RM(nn.Module):
    def __init__(self, base, vocab_size=32000):
        super(Llama3_2_RM, self).__init__()
        self.base_model = base
        self.LN = nn.Linear(vocab_size, 1)

    def forward(self, inputs):
        outputs = self.base_model(**inputs, max_new_tokens=1000).logits[:, -1, :]
        value_outputs = self.LN(outputs)
        return value_outputs.squeeze(dim=1)


# Load training set, validation set, and test set data
train_js = f'result/{dataset}/train.json'
val_js = f'result/{dataset}/validate.json'


def read_json(source):
    with open(source, 'r', encoding='utf-8') as f:
        json_list = json.load(f)
    return json_list


train_json = read_json(train_js)  # This section uses a CSV file as an example to describe how to load data
val_json = read_json(val_js)

# Create a custom dataset
train_dataset = MyDataset(train_json, processor,'train')
val_dataset = MyDataset(val_json, processor,'validate')

# Create data loaders
batch_size = 3  # Set batch size
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Set device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, '\n')
vocab_size = model.config.vocab_size
print(vocab_size)
VM = Llama3_2_RM(model, vocab_size)
VM.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = AdamW(VM.parameters(), lr=1e-6)
num_epochs = 2
# Training and validation loop
best_val_loss = 10000000
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    print(f"{epoch}/{num_epochs} training")
    # Training
    VM.train()
    train_loss = 0.0
    for batch in tqdm(train_dataloader):
        inputs = batch['inputs'].to(device)
        labels = batch['label'].to(dtype=torch.float32).to(device)

        optimizer.zero_grad()
        outputs = VM(inputs=inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    # Validation
    VM.eval()
    val_loss = 0.0
    val_labels = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            inputs = batch['inputs'].to(device)
            labels = batch['label'].to(dtype=torch.float32).to(device)
            outputs = VM(inputs=inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_labels.extend(labels.tolist())

    avg_val_loss = val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(VM.state_dict(), "models/Llama3_2/VM_best_checkpoint.pt")

print("Training complete!")

# # Load the best model for inference
# best_model = Llama3_2_RM(model, vocab_size)
# best_model.load_state_dict(torch.load("records/Mistral/VM_best_checkpoint.pt"))
# best_model.to(device)
# best_model.eval()
#
# # Perform inference
# test_preds = []
# test_labels = []
# with torch.no_grad():
#     for batch in tqdm(test_dataloader):
#         inputs = batch['inputs'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['label'].to(dtype=torch.float32).to(device)
#         outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
#         test_preds.extend(outputs.tolist())
#         test_labels.extend(labels.tolist())
#     print("Inference results:")
#     for i in range(len(test_preds)):
#         print(f"Sample {i + 1}: Predicted score {test_preds[i]}, Actual score {test_labels[i]}, Truncated score {min(max(test_preds[i], 0), 1)}")
#
# cnt = 0
# for i in range(len(test_preds)):
#     if abs(min(max(test_preds[i], 0), 1) - test_labels[i]) <= 0.1:
#         cnt += 1
# test_acc = cnt / len(test_preds)
# print(f"Test accuracy: {test_acc:.4f}")