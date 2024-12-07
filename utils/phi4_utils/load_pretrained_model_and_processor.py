from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import torch

def load_pretrained_model(device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
                _attn_implementation='flash_attention_2',):
    path = "microsoft/Phi-4-multimodal-instruct"
    model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                # if you do not use Ampere or later GPUs, change attention to "eager"
                _attn_implementation='flash_attention_2',
            ).cuda()
    return model

def load_pretrained_processor(trust_remote_code=True, use_fast=False):
    path = "microsoft/Phi-4-multimodal-instruct"
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    return processor
