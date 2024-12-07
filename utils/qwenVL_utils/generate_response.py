import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import GenerationConfig
from qwen_vl_utils import process_vision_info

def resize_image_if_needed(img, max_size=512):
    """
    等比缩放图片，如果宽或高超过 max_size，就缩放到最长边为 max_size，保持宽高比不变。
    :param img: PIL.Image 对象
    :param max_size: 允许的最大边长
    :return: 缩放后的 PIL.Image 对象
    """
    width, height = img.size
    # 如果最长边大于 max_size，则需要进行缩放
    if max(width, height) > max_size:
        # 等比缩放比例
        scale_ratio = max_size / float(max(width, height))
        new_width = int(width * scale_ratio)
        new_height = int(height * scale_ratio)
        # 进行缩放
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img


def generate_response(processor, model, prompt, images_path, max_new_tokens=1024, do_sample=True, temperature=0.3):
    # Part 1: Image Processing
    # print("\n--- IMAGE PROCESSING ---")
    # print(f'>>> Prompt\n{prompt}')

    # Download and open image
    prompt[0]['content'][0]['image'] = images_path

    # Generate response
    text = processor.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(prompt)
    image_inputs = [resize_image_if_needed(image_inputs[0])]
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return response[0]