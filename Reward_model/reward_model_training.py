import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AdamW
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator, DeepSpeedPlugin
from utils.phi_vision_utils.load_pretrained_model_and_processor import load_pretrained_processor
from PIL import Image
import wandb
from Reward_model.reward_model_dataset import MyDataset
from Reward_model.reward_model import Phi_Vision_RM

os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
path = "/home/qi/python_project/multimodal_reasoning"
os.chdir(path)

max_length = 900
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


# Load training set, validation set, and test set data
train_js = f'Reward_model/data/train.json'
val_js = f'Reward_model/data/validation.json'
wandb.init(
     project="PRM_phi3.5_MMPR"
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
learning_rate = 5e-7
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Set device and model
device = accelerator.device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VM = Phi_Vision_RM(model_id)
VM.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = AdamW(VM.parameters(), lr=learning_rate)
VM, optimizer, train_dataloader, val_dataloader= accelerator.prepare(
    VM, optimizer, train_dataloader, val_dataloader
)
num_epochs = 10
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
            "loss": loss
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
            "train_loss": avg_train_loss,
            "valid_loss": avg_val_loss
        })

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ")

        # Save best model
        if avg_val_loss < best_val_loss and accelerator.is_main_process:
            best_val_loss = avg_val_loss
            unwrapped_model = accelerator.unwrap_model(VM)  # 拿到未封装的原始Phi_Vision_RM实例
            unwrapped_model.base_model.save_pretrained("Reward_model/weights/base_model", safe_serialization=False)
            torch.save(unwrapped_model.LN.state_dict(), "Reward_model/weights/LN.pt")
        print(f"{epoch}/{num_epochs} training")

print("Training complete!")