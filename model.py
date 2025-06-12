from transformers import Qwen2VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# # Define LoRA configuration
# lora_config = LoraConfig(
#     r=8,             # Rank for dimensionality reduction (higher = better performance but more compute)
#     lora_alpha=16,   # Scaling factor for LoRA weights
#     target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA to (GPT example)
#     lora_dropout=0.1,  # Dropout probability for LoRA layers
#     bias="none"      # Whether to apply LoRA to biases ("none", "all", or "lora_only")
# )

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
        # Linear layer mapping from vocabulary size to single scalar reward.
        self.LN = nn.Linear(self.base_model.config.vocab_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        # Passes multimodal inputs through the base Qwen2-VL model to get logits.
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                                  pixel_values = pixel_values, image_grid_thw = image_grid_thw)
        # Passes multimodal inputs through the base Qwen2-VL model to get logits.
        # [:, -1, :]: Takes logits of the final token position.
        outputs = outputs.logits[:, -1, :].to(dtype=torch.float)
        # print(outputs)
        # Maps logits to scalar reward using linear layer.
        value_outputs = self.LN(outputs)
        # Applies sigmoid to get probability in [0,1] range.
        value_outputs = self.sigmoid(value_outputs)
        # print(value_outputs)
        # Removes dimension to return shape [batch_size] instead of [batch_size, 1].
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


# # Define LoRA configuration
# lora_config = LoraConfig(
#     r=8,             # Rank for dimensionality reduction (higher = better performance but more compute)
#     lora_alpha=16,   # Scaling factor for LoRA weights
#     target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA to (GPT example)
#     lora_dropout=0.1,  # Dropout probability for LoRA layers
#     bias="none"      # Whether to apply LoRA to biases ("none", "all", or "lora_only")
# )

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
        # Linear layer mapping from vocabulary size to single scalar reward.
        self.LN = nn.Linear(self.base_model.config.vocab_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        # Passes multimodal inputs through the base Qwen2-VL model to get logits.
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                                  pixel_values = pixel_values, image_grid_thw = image_grid_thw)
        # Passes multimodal inputs through the base Qwen2-VL model to get logits.
        # [:, -1, :]: Takes logits of the final token position.
        outputs = outputs.logits[:, -1, :].to(dtype=torch.float)
        # print(outputs)
        # Maps logits to scalar reward using linear layer.
        value_outputs = self.LN(outputs)
        # Applies sigmoid to get probability in [0,1] range.
        value_outputs = self.sigmoid(value_outputs)
        # print(value_outputs)
        # Removes dimension to return shape [batch_size] instead of [batch_size, 1].
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
    

class DomainTable(nn.Module):
    def __init__(self, domain_to_idx):
        """
        Args:
            domain_to_idx (dict):
                Mapping from domain strings to integer indices, e.g., {"domain_a": 0, "domain_b": 1}.
        """
        super(DomainTable, self).__init__()
        self.domain_to_idx = domain_to_idx  # Maps domain names (like "AI2D", "M3CoT") to indices.
        self.num_domains = len(domain_to_idx)   # Number of unique domains.

        # Creates learnable parameters for domain weights,
        # (initialized to zero, will be optimized during bi-level optimization).
        self.raw_weights = nn.Parameter(torch.zeros(self.num_domains))

    def forward(self, domain_strings, x):
        """
        Args:
            domain_strings (list[str] or tuple[str]):
                Domain names for each sample in the batch. Length should match x's batch_size.
            x (torch.Tensor):
                Input tensor of shape (batch_size, 1), containing a single value per sample.

        Returns:
            torch.Tensor:
                Output tensor of same shape (batch_size, 1), where each element is the original input
                multiplied by its corresponding domain weight.
        """
        # Apply softplus activation to ensure weights are positive.
        # Softplus(x) = log(1 + exp(x)) ensures output > 0.
        positive_weights = torch.nn.functional.softplus(self.raw_weights)

        # Normalize weights by their mean to maintain scale.
        # Ensures the average weight remains around 1.0.
        # This stabilizes training and makes weights interpretable.
        mean_weights = positive_weights.mean()
        normalized_weights = positive_weights / mean_weights

        # Convert domain strings to indices matching batch order.
        # Maps domain names to their corresponding indices.
        idxes = [self.domain_to_idx[d] for d in domain_strings]
        # Creates tensor of indices for batch lookup.
        idxes = torch.tensor(idxes, dtype=torch.long, device=x.device)  # [batch_size]

        # Retrieve domain weights for each sample in the batch [batch_size].
        # Uses advanced indexing to get weight for each sample's domain.
        domain_weights = normalized_weights[idxes]

        # Reshape weights to match input tensor dimensions [batch_size, 1].
        domain_weights = domain_weights.view(-1, 1)

        # Element-wise multiplication: each input value multiplied by its domain weight.
        # Implements the domain reweighting in the lower-level optimization.
        out = x * domain_weights
        return out
    

class Math_DomainTable(nn.Module):
    def __init__(self):
        super().__init__()
        self.domain_to_idx = {
            "gsm8k": 0,
            "math": 1,
            "aime_train": 2,
            "imo": 3,
            "mathqa": 4,
            "theorem_qa": 5,
            "openmathinstruct_pot": 6,
            "qwen_math_synthetic": 7  # Additional Qwen math data
        }
        
        # Initialize with math-focused weights
        initial_weights = torch.tensor([
            0.6,  # GSM8K (elementary)
            1.3,  # MATH (competition)
            1.6,  # AIME (very hard)
            1.4,  # IMO (olympiad)
            0.9,  # MathQA (word problems)
            1.2,  # TheoremQA (proof-based)
            1.5,  # OpenMathInstruct PoT
            1.1   # Qwen synthetic data
        ])
        
        self.raw_weights = nn.Parameter(initial_weights)
    
    def forward(self, domain_strings, rewards):
        # Apply softplus and normalize
        weights = torch.softplus(self.raw_weights)
        normalized_weights = weights / weights.mean()
        
        # Get domain weights for batch
        domain_indices = [self.domain_to_idx[d] for d in domain_strings]
        domain_indices = torch.tensor(domain_indices, dtype=torch.long, device=rewards.device)
        batch_weights = normalized_weights[domain_indices]
        
        # Apply weighting - ensure proper shape matching
        if rewards.dim() == 1:
            return rewards * batch_weights
        else:
            return rewards * batch_weights.view(-1, 1)