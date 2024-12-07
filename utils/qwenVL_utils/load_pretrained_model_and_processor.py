from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch

def load_pretrained_model(device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
                _attn_implementation='flash_attention_2',):
    path = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, torch_dtype="auto", device_map="auto"
    )
    return model

def load_pretrained_processor(trust_remote_code=True, use_fast=False):
    path = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(path)
    return processor
