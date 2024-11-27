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

os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

max_length = 900
dataset = "ScienceQA"
model_id = "microsoft/Phi-3.5-vision-instruct"

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
)

# Load the pre-trained Llama3.2
deepspeed = DeepSpeedPlugin(zero_stage=2,gradient_accumulation_steps=4)
accelerator = Accelerator(
    mixed_precision='bf16',
    deepspeed_plugin=deepspeed,
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

        label = self.data_js[idx]['label']

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'image_sizes': inputs['image_sizes'].squeeze(),
            'label': label
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
        outputs = self.base_model(input_ids=input_ids,attention_mask=attention_mask, pixel_values = pixel_values, image_sizes = image_sizes).logits[:, -1, :].to(dtype=torch.bfloat16)
        # print(outputs)
        value_outputs = self.LN(outputs)
        # print(value_outputs)
        return value_outputs.squeeze(dim=1)


# Load training set, validation set, and test set data
train_js = f'results/{dataset}/phi3_5/reward_train.json'
val_js = f'results/{dataset}/phi3_5/reward_valid.json'
wandb.init(
     project="PRM_phi3.5"
    )


def read_json(source):
    with open(source, 'r', encoding='utf-8') as f:
        json_list = json.load(f)
    return json_list


train_json = read_json(train_js)
val_json = read_json(val_js)

# Create a custom dataset
train_dataset = MyDataset(train_json, processor,'train')
val_dataset = MyDataset( val_json, processor,'train')

# Create data loaders
batch_size = 1  # Set batch size
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Set device and model
# device = accelerator.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VM = Phi3_5_RM(model_id)
VM.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = AdamW(VM.parameters(), lr=1e-6)
VM, optimizer, train_dataloader, val_dataloader= accelerator.prepare(
    VM, optimizer, train_dataloader, val_dataloader
)
num_epochs = 5
# Training and validation loop
best_val_loss = 10000000
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    # Training
    VM.train()
    train_loss = 0.0
    for batch in tqdm(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        image_sizes = batch['image_sizes'].to(device)
        labels = batch['label'].to(dtype=torch.bfloat16).to(device)

        optimizer.zero_grad()
        outputs = VM(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_sizes=image_sizes)
        loss = criterion(outputs, labels)
        # print(loss)
        wandb.log({
            "train_loss": loss
        })
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(VM.parameters(), 1.0)
        # loss.backward()
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            image_sizes = batch['image_sizes'].to(device)
            labels = batch['label'].to(dtype=torch.bfloat16).to(device)

            optimizer.zero_grad()
            outputs = VM(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
                         image_sizes=image_sizes)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_labels.extend(labels.tolist())

        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        wandb.log({
            "valid_loss": avg_val_loss
        })

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ")

        # Save best model
        if avg_val_loss < best_val_loss and accelerator.is_main_process:
            best_val_loss = avg_val_loss
            accelerator.save(VM.state_dict(), "models/phi3_5/VM_best_checkpoint.pth")
        print(f"{epoch}/{num_epochs} training")

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