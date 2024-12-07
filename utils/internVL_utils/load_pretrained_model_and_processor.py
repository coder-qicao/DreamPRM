from transformers import AutoTokenizer, AutoModel
import torch

def load_pretrained_model(torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True):
    path = "OpenGVLab/InternVL2_5-4B"
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        use_flash_attn=use_flash_attn,
        trust_remote_code=trust_remote_code).cuda()
    return model

def load_pretrained_tokenizer(trust_remote_code=True, use_fast=False):
    path = "OpenGVLab/InternVL2_5-4B"
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast)
    return tokenizer

def load_pretrained_model_MPO(torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True):
    path = "OpenGVLab/InternVL2_5-8B-MPO"
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        use_flash_attn=use_flash_attn,
        trust_remote_code=trust_remote_code).cuda()
    return model

def load_pretrained_tokenizer_MPO(trust_remote_code=True, use_fast=False):
    path = "OpenGVLab/InternVL2_5-8B-MPO"
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast)
    return tokenizer
