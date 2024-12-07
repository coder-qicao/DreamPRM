import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import GenerationConfig


def generate_response(processor, model, prompt, images_path, max_new_tokens=1024, do_sample=True, temperature=0.3):
    # Load generation config
    model_path = "microsoft/Phi-4-multimodal-instruct"
    generation_config = GenerationConfig.from_pretrained(model_path)

    # Part 1: Image Processing
    # print("\n--- IMAGE PROCESSING ---")
    # print(f'>>> Prompt\n{prompt}')

    # Download and open image
    image = Image.open(images_path)
    inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')

    # Generate response
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        generation_config=generation_config,
        do_sample=do_sample,
        temperature=temperature
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response