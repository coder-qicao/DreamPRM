import torch
from Reward_model.reward_model import Phi_Vision_RM

def load_Phi_Vision_RM(model_id = "models/phi3_5/VM_base_best_checkpoint",
                      LN_id = "models/phi3_5/VM_LN_best_checkpoint.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = Phi_Vision_RM(model_id)
    best_model.LN.load_state_dict(torch.load(LN_id))
    best_model.to(device)
    return best_model

def generate_reward_model_input(prompt, image, processor):
    # find question and prompt
    messages = [
        {"role": "user", "content": prompt},
    ]
    prompt_answer = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(prompt_answer, image, return_tensors="pt")

    return {
        'input_ids': inputs['input_ids'].squeeze(),
        'attention_mask': inputs['attention_mask'].squeeze(),
        'pixel_values': inputs['pixel_values'].squeeze(),
        'image_sizes': inputs['image_sizes'].squeeze(),
    }
