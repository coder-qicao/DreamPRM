import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
device = "cuda:0"
model_name = "Skywork/Skywork-Reward-Llama-3.1-8B"
rm = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="eager",
    num_labels=1,
)
rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

part = "output_claude_2shot_solution_use_caption_ocr"
path = "results/MathVista/"
result = {}
true_score = []
false_score = []
with open(f"{path}{part}.json", "r", encoding='utf-8') as f:
    data = json.load(f)

for i in range(1000):
    name = str(i+1)
    if 'caption' in data[name]:
        prompt = data[name]["caption"] + data[name]["query"]
    else:
        prompt = data[name]["query"]
    response = data[name]["response"]

    conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]

    # Format and tokenize the conversations
    conv_formatted = rm_tokenizer.apply_chat_template(conv, tokenize=False)
    conv_tokenized = rm_tokenizer(conv_formatted, return_tensors="pt").to(device)

    # Get the reward scores
    with torch.no_grad():
        score = rm(**conv_tokenized).logits[0][0].item()
    result[name] = {"score": score, "True or False":data[name]['true_false']}
    print(f"Score for response {name}: {score}")
    print(f"True or False: {data[name]['true_false']}")
    if data[name]['true_false']:
        true_score.append(score)
    else:
        false_score.append(score)

print(sum(true_score) / len(true_score))
print(sum(false_score) / len(false_score))
json.dump(result, open(f"{path}{part}_result.json", "w", encoding='utf-8'))

# Output:
# Score for response 1: 12.625
# Score for response 2: -15.25
