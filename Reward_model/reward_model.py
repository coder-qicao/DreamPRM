import torch.nn as nn
from transformers import AutoModelForCausalLM
import torch

class Phi_Vision_RM(nn.Module):
    def __init__(self, model_id, trust_remote_code=True,
            torch_dtype="bfloat16",
            _attn_implementation='flash_attention_2'):
        super(Phi_Vision_RM, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            _attn_implementation=_attn_implementation
        )
        self.LN = nn.Linear(self.base_model.config.vocab_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, pixel_values, image_sizes):
        outputs = self.base_model(input_ids=input_ids,attention_mask=attention_mask, pixel_values = pixel_values, image_sizes = image_sizes).logits[:, -1, :].to(dtype=torch.bfloat16)
        # print(outputs)
        value_outputs = self.LN(outputs)
        value_outputs = self.sigmoid(value_outputs)
        # print(value_outputs)
        return value_outputs.squeeze(dim=1)