import os
from PIL import Image

def zero_shot_prompt_building_single_image(data, image_path, processor, tokenize=False, add_generation_prompt=True):
    i = data # i has the format {'id':, 'input':,}
    placeholder = f"<|image_1|>\n"

    if os.path.isfile(image_path):
        image = [Image.open(image_path)]
        text = i['input']
        query = text + "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end. Let's answer step by step."
        messages = [
            {"role": "user", "content": placeholder + query},
        ]
    else:
        image = None
        text = i['input']
        query = text + "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end. Let's answer step by step."
        messages = [
            {"role": "user", "content": query},
        ]

    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt
    )
    return prompt, image

def generate_function_input_single_image(data, dataset, part):
    input = data["input"]
    image_path = f"datasets/{dataset}/images/{part}/{data['id']}.png"
    return input, image_path