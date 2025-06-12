from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

# Set the specific download directory
download_dir = "/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/weights/Qwen2.5-Math-PRM-7B"

# Download the tokenizer (this should work without PyTorch)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-PRM-7B")
tokenizer.save_pretrained(download_dir)

# Download the model files directly without loading them
snapshot_download(
    repo_id="Qwen/Qwen2.5-Math-PRM-7B",
    local_dir=download_dir,
    local_dir_use_symlinks=False
)

print(f"Model and tokenizer successfully downloaded to: {download_dir}")