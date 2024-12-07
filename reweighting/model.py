import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

# 定义 LoRA 配置
lora_config = LoraConfig(
    r=8,             # 降维后的秩 (越大效果越好，但计算需求也更高)
    lora_alpha=16,   # LoRA 缩放因子
    target_modules=["q_proj", "v_proj"],  # 要应用 LoRA 的模块 (以 GPT 为例)
    lora_dropout=0.1,  # dropout 概率
    bias="none"      # 是否对 bias 应用 LoRA ("none", "all", or "lora_only")
)

class QwenVL_RM(nn.Module):
    def __init__(self, device, model_path="Qwen/Qwen2-VL-2B-Instruct"):
        super(QwenVL_RM, self).__init__()
        self.base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
        # self.lora_model = get_peft_model(base_model, lora_config)
        self.LN = nn.Linear(self.base_model.config.vocab_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                                  pixel_values = pixel_values, image_grid_thw = image_grid_thw)
        outputs = outputs.logits[:, -1, :].to(dtype=torch.float)
        # print(outputs)
        value_outputs = self.LN(outputs)
        value_outputs = self.sigmoid(value_outputs)
        # print(value_outputs)
        return value_outputs.squeeze(dim=1)

class Llava_RM(nn.Module):
    def __init__(self, device):
        super(Llava_RM, self).__init__()
        self.base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attention_2=True
        ).to(0)
        # self.lora_model = get_peft_model(base_model, lora_config)
        self.LN = nn.Linear(self.base_model.vocab_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, pixel_values, image_sizes):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                                  pixel_values = pixel_values, image_sizes = image_sizes)
        outputs = outputs.logits[:, -1, :].to(dtype=torch.float)
        # print(outputs)
        value_outputs = self.LN(outputs)
        value_outputs = self.sigmoid(value_outputs)
        # print(value_outputs)
        return value_outputs.squeeze(dim=1)


import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainTable(nn.Module):
    def __init__(self, domain_to_idx):
        """
        Args:
            domain_to_idx (dict):
                字符串 -> 整数索引 的映射，例如 {"domain_a": 0, "domain_b": 1}。
        """
        super(DomainTable, self).__init__()
        self.domain_to_idx = domain_to_idx
        self.num_domains = len(domain_to_idx)

        # 创建可学习的 raw_weights
        self.raw_weights = nn.Parameter(torch.zeros(self.num_domains))  # 初始为0

    def forward(self, domain_strings, x):
        """
        Args:
            domain_strings (list[str] or tuple[str]):
                每个样本对应的 domain 名称，长度与 x 的 batch_size 相同。
            x (torch.Tensor):
                形状为 (batch_size, 1)，表示每个样本一个数值。

        Returns:
            torch.Tensor:
                同形状 (batch_size, 1) 的张量，每个元素等于原输入乘以对应的 domain 权重。
        """
        positive_weights = torch.nn.functional.softplus(self.raw_weights)
        mean_weights =  positive_weights.mean()
        normalized_weights = positive_weights / mean_weights

        # 将字符串 domain 转成索引，保证顺序与 batch 对应
        idxes = [self.domain_to_idx[d] for d in domain_strings]
        idxes = torch.tensor(idxes, dtype=torch.long, device=x.device)  # [batch_size]

        # 取出每个 domain 的 scalar 权重，形状 [batch_size]
        domain_weights = normalized_weights[idxes]

        # 将 domain_weights reshape 为 [batch_size, 1]，与 x 对齐后做逐元素乘法
        domain_weights = domain_weights.view(-1, 1)

        # x 形状是 (batch_size, 1)，元素逐一乘以对应权重
        out = x * domain_weights
        return out
