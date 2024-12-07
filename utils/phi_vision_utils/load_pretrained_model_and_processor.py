from transformers import AutoModelForCausalLM, AutoProcessor

def load_pretrained_model(device_map="cuda",
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation='flash_attention_2'):
    model_id = "microsoft/Phi-3.5-vision-instruct"
    # Note: set _attn_implementation='eager' if you don't have flash_attn installed
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        _attn_implementation=_attn_implementation
    )
    return model

def load_pretrained_processor(trust_remote_code=True,
        num_crops=16):
    model_id = "microsoft/Phi-3.5-vision-instruct"
    # Note: to achieve the best performances to set num_crops=4 for multi-frame and num_crops=16 for single-frame
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        num_crops=num_crops
    )
    return processor