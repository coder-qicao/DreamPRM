import copy
import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from qwen_vl_utils import process_vision_info
import json
from transformers import AutoProcessor
from PIL import Image
import re


def split_step(s_id, response):
    s = f"Step {s_id}"
    s_next = f"Step {s_id+1}"
    if s_next in response:
        assistant = response.split(s_next)[0]
    elif "Final answer" in response and s in response:
        assistant = response.split("Final answer")[0]
    else:
        assistant = ""
    return assistant


def find_max_step(response):
    """
    Find the maximum step number in a response string containing steps.

    Args:
        response: String containing steps in formats like "Step 1: ...", "Step 2: ...", etc.

    Returns:
        Integer representing the highest step number found. Returns 0 if no steps are found.
    """
    # Find all occurrences of step patterns (case-insensitive)
    # Matches: "Step 1", "STEP 2", "step3", "Step: 4", etc.
    step_numbers = re.findall(r'Step[\s:]*(\d+)', response, re.IGNORECASE)

    # Return 0 if no step numbers found
    if not step_numbers:
        return 0

    # Convert found numbers from strings to integers
    step_numbers = [int(num) for num in step_numbers]

    # Return the maximum step number
    return max(step_numbers)


def read_json(source):
    with open(source, 'r', encoding='utf-8') as f:
        json_list = json.load(f)
    return json_list


def resize_image_if_needed(img, max_size=512):
    """
    Resize an image proportionally if either width or height exceeds max_size.
    Maintains the original aspect ratio while scaling down the longest side to max_size.

    :param img: PIL.Image object to be resized
    :param max_size: Maximum allowed length for the longest side (default: 512)
    :return: Resized PIL.Image object
    """
    width, height = img.size
    # Check if the longest dimension exceeds max_size
    if max(width, height) > max_size:
        # Calculate scaling ratio while maintaining aspect ratio
        scale_ratio = max_size / float(max(width, height))
        new_width = int(width * scale_ratio)
        new_height = int(height * scale_ratio)
        # Resize image using LANCZOS resampling for high quality
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img

class MyDataset_QwenVL(Dataset):
    def __init__(self, data_js, processor):
        self.data_js = data_js
        self.processor = processor

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        # find question and prompt
        i = self.data_js[idx]['id']
        prompt = self.data_js[idx]['input']
        add = self.data_js[idx]['add']
        image_path = self.data_js[idx]['image_path']
        prompt = prompt + "\n\n" + add
        label = self.data_js[idx]['accuracy']
        dset = self.data_js[idx]['dataset']

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        image_inputs = [resize_image_if_needed(image_inputs[0])]
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'image_grid_thw': inputs['image_grid_thw'].squeeze(),
            'label': label,
            'dataset': dset
        }


class MyDataset_Llava(Dataset):
    def __init__(self, data_js, processor):
        self.data_js = data_js
        self.processor = processor

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        # find question and prompt
        i = self.data_js[idx]['id']
        prompt = self.data_js[idx]['input']
        add = self.data_js[idx]['add']
        image_path = self.data_js[idx]['image_path']
        prompt = prompt + "\n\n" + add
        label = self.data_js[idx]['accuracy']
        dset = self.data_js[idx]['dataset']

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        raw_image = Image.open(image_path)
        inputs = self.processor(images=raw_image, text=text, return_tensors='pt').to(0, torch.bfloat16)

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'image_sizes': inputs['image_sizes'].squeeze(),
            'label': label,
            'dataset': dset
        }


class MyDataset_QwenMath(Dataset):
    """Dataset class for QwenMath text-only mathematical reasoning"""
    def __init__(self, data_js, tokenizer):
        self.data_js = data_js
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        item = self.data_js[idx]
        
        # Extract data fields
        problem = item.get('problem', '')
        prompt = item.get('prompt', '')
        label = item.get('accuracy', item.get('label', 0.0))
        dset = item.get('domain', item.get('dataset', 'unknown'))
        
        # Create input text - use prompt if available, otherwise create from problem
        if prompt:
            input_text = prompt
        else:
            input_text = f"""<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant specialized in mathematics.<|im_end|>
<|im_start|>user
Solve this mathematical problem step by step:

{problem}<|im_end|>
<|im_start|>assistant
I'll solve this step by step.

"""

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': float(label),
            'dataset': dset
        }
    

class MyMetaDataset_QwenVL(Dataset):
    def __init__(self, data_js, processor):
        self.data_js = data_js
        self.processor = processor

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        # find question and prompt
        input = self.data_js[idx]['input']
        image_path = self.data_js[idx]['image_path']
        label = self.data_js[idx]['true_false']

        r_dict = {}
        step_num = find_max_step(input)
        for index in range(step_num):
            step = split_step(index+1, input)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": step},
                    ],
                }
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            image_inputs = [resize_image_if_needed(image_inputs[0])]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            r_dict[f"{index+1}"] = {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'pixel_values': inputs['pixel_values'].squeeze(),
                'image_grid_thw': inputs['image_grid_thw'].squeeze(),
            }
        r_dict["labels"] = torch.tensor(label).to(dtype=torch.float)
        return r_dict


class MyMetaDataset_Llava(Dataset):
    def __init__(self, data_js, processor, step_num = 5):
        self.data_js = data_js
        self.processor = processor
        self.step_num = step_num

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        # find question and prompt
        input = self.data_js[idx]['input']
        image_path = self.data_js[idx]['image_path']
        label = self.data_js[idx]['true_false']
        r_dict = {}
        for index in range(self.step_num):
            step = split_step(index+1, input)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": step},
                        {"type": "image"},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            raw_image = Image.open(image_path)
            inputs = self.processor(images=raw_image, text=text, return_tensors='pt').to(0, torch.bfloat16)
            r_dict[f"{index+1}"] = {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'pixel_values': inputs['pixel_values'].squeeze(),
                'image_sizes': inputs['image_sizes'].squeeze(),
            }
        r_dict["labels"] = torch.tensor(label).to(dtype=torch.float)
        return r_dict


class MyMetaDataset_QwenMath(Dataset):
    """Meta dataset for QwenMath step-by-step evaluation"""
    def __init__(self, data_js, tokenizer):
        self.data_js = data_js
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        item = self.data_js[idx]
        
        # Extract data
        input_text = item.get('input', item.get('full_solution', ''))
        problem = item.get('problem', '')
        label = item.get('true_false', [])
        
        r_dict = {}
        
        # Find steps in the input
        step_num = find_max_step(input_text)
        
        # If no steps found but we have a steps list, use that
        if step_num == 0 and 'steps' in item:
            steps_list = item['steps']
            step_num = len(steps_list)
        
        # Process each step
        for index in range(step_num):
            if 'steps' in item and index < len(item['steps']):
                # Use pre-extracted steps
                step_content = item['steps'][index]
            else:
                # Extract step from input text
                step_content = split_step(index+1, input_text)
            
            if not step_content.strip():
                continue
                
            # Create step evaluation prompt
            step_prompt = f"""<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant specialized in mathematics.<|im_end|>
<|im_start|>user
Problem: {problem}

Evaluate this reasoning step:
{step_content}

Rate the quality of this step (0.0 to 1.0):<|im_end|>
<|im_start|>assistant
"""
            
            # Tokenize
            inputs = self.tokenizer(
                step_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            )
            
            r_dict[f"{index+1}"] = {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
            }
        
        # Handle labels
        if isinstance(label, list) and len(label) > 0:
            r_dict["labels"] = torch.tensor(label, dtype=torch.float)
        elif isinstance(label, (int, float)):
            # Single label for all steps
            r_dict["labels"] = torch.tensor([label] * step_num, dtype=torch.float)
        else:
            # Default labels
            r_dict["labels"] = torch.tensor([1.0] * step_num, dtype=torch.float)
        
        return r_dict


def build_dataloader(
        processor_path,
        train_json_file,
        meta_json_file,
        train_batch_size,
        meta_batch_size,
):
    processor = AutoProcessor.from_pretrained(processor_path)
    train_dataset = MyDataset_QwenVL(read_json(train_json_file), processor)
    meta_dataset = MyMetaDataset_QwenVL(read_json(meta_json_file), processor)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    meta_dataloader = DataLoader(meta_dataset, batch_size=meta_batch_size, shuffle=True)

    return train_dataloader, meta_dataloader


def build_qwen_math_dataloader(
        tokenizer_path,
        train_json_file,
        meta_json_file,
        train_batch_size,
        meta_batch_size,
):
    """Build dataloaders for QwenMath training"""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        pad_token_id=151643
    )
    
    train_dataset = MyDataset_QwenMath(read_json(train_json_file), tokenizer)
    meta_dataset = MyMetaDataset_QwenMath(read_json(meta_json_file), tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    meta_dataloader = DataLoader(meta_dataset, batch_size=meta_batch_size, shuffle=True)

    return train_dataloader, meta_dataloader


class QwenMathDataProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Math-7B-Instruct",
            trust_remote_code=True
        )
    
    def create_qwen_prompt(self, problem, solution_type="step_by_step"):
        """Create Qwen-compatible prompts"""
        
        if solution_type == "step_by_step":
            return f"""<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant specialized in mathematics.<|im_end|>
<|im_start|>user
Solve this mathematical problem step by step:

{problem}<|im_end|>
<|im_start|>assistant
I'll solve this step by step.

"""
        
        elif solution_type == "pot":
            return f"""<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant specialized in mathematics. You can use Python code to solve problems.<|im_end|>
<|im_start|>user
Solve this mathematical problem using both reasoning and Python code:

{problem}<|im_end|>
<|im_start|>assistant
I'll solve this using mathematical reasoning and Python code.

"""
    
    def process_openmathinstruct_for_qwen(self, examples, max_examples=10000):
        """Process OpenMathInstruct data for Qwen format"""
        processed = []
        
        for i, example in enumerate(examples[:max_examples]):
            if i % 1000 == 0:
                print(f"Processed {i} examples...")
            
            problem = example.get('problem', example.get('question', ''))
            response = example.get('response', example.get('solution', ''))
            
            # Create Qwen-compatible format
            qwen_prompt = self.create_qwen_prompt(problem, "pot")
            
            # Extract reasoning steps
            steps = self.extract_reasoning_steps(response)
            
            if len(steps) >= 2:  # At least 2 steps for meaningful training
                processed.append({
                    'problem': problem,
                    'prompt': qwen_prompt,
                    'full_solution': response,
                    'steps': steps,
                    'domain': 'openmathinstruct_pot',
                    'ground_truth': self.extract_answer(response),
                    'accuracy': 1.0,  # Assume OpenMathInstruct solutions are correct
                    'dataset': 'openmathinstruct_pot'
                })
        
        return processed
    
    def extract_reasoning_steps(self, solution):
        """Extract individual reasoning steps from solution"""
        steps = []
        
        # Split by common step indicators
        step_patterns = [
            r'Step \d+:',
            r'\d+\.',
            r'First,',
            r'Next,',
            r'Then,',
            r'Finally,',
            r'Therefore,'
        ]
        
        # Simple splitting by sentences/paragraphs
        sentences = solution.split('\n\n')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Filter out very short fragments
                steps.append(sentence)
        
        return steps
    
    def extract_answer(self, solution):
        """Extract numerical answer from solution"""
        import re
        
        # Common answer patterns
        patterns = [
            r'(?:answer|result|solution)\s*(?:is|=)\s*(\d+(?:\.\d+)?)',
            r'(?:therefore|thus|so),?\s*(?:the answer is\s*)?(\d+(?:\.\d+)?)',
            r'=\s*(\d+(?:\.\d+)?)(?:\s|$)',
            r'(\d+(?:\.\d+)?)\s*(?:is the (?:answer|result))'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def convert_to_training_format(self, processed_data):
        """Convert processed data to training format compatible with existing structure"""
        training_data = []
        
        for item in processed_data:
            training_data.append({
                'problem': item['problem'],
                'prompt': item['prompt'],
                'accuracy': item.get('accuracy', 1.0),
                'dataset': item['domain'],
                'ground_truth': item.get('ground_truth'),
                'steps': item.get('steps', [])
            })
        
        return training_data
    
    def convert_to_meta_format(self, processed_data):
        """Convert processed data to meta format for step evaluation"""
        meta_data = []
        
        for item in processed_data:
            if 'steps' in item and len(item['steps']) > 0:
                # Create step-wise labels (assume all steps are correct for OpenMathInstruct)
                step_labels = [1.0] * len(item['steps'])
                
                meta_data.append({
                    'problem': item['problem'],
                    'input': item['full_solution'],
                    'steps': item['steps'],
                    'true_false': step_labels,
                    'domain': item['domain']
                })
        
        return meta_data


# Load and process data
def load_qwen_training_data():
    """Load all training data in Qwen format"""
    processor = QwenMathDataProcessor()
    
    # Load OpenMathInstruct-1
    try:
        from datasets import load_dataset
        openmathinstruct = load_dataset("nvidia/OpenMathInstruct-1", split="train", streaming=True)
        
        # Process data
        pot_data = processor.process_openmathinstruct_for_qwen(
            list(openmathinstruct.take(50000)),  # Take first 50k examples
            max_examples=10000
        )
        
        # Convert to training format
        training_data = processor.convert_to_training_format(pot_data)
        meta_data = processor.convert_to_meta_format(pot_data)
        
        return training_data, meta_data
        
    except ImportError:
        print("datasets library not available, returning empty data")
        return [], []
    except Exception as e:
        print(f"Error loading OpenMathInstruct data: {e}")
        return [], []


def process_domain_for_qwen(domain_name, raw_data, processor):
    """Process traditional math domains for Qwen"""
    processed = []
    
    for item in raw_data[:2000]:  # Limit to avoid memory issues
        problem = item.get('problem', item.get('question', ''))
        answer = item.get('answer', item.get('solution', ''))
        
        # Create Qwen prompt
        qwen_prompt = processor.create_qwen_prompt(problem, "step_by_step")
        
        # Generate solution using base Qwen model (if available)
        # Or use existing solutions
        steps = processor.extract_reasoning_steps(str(answer))
        
        if steps:
            processed.append({
                'problem': problem,
                'prompt': qwen_prompt,
                'steps': steps,
                'domain': domain_name,
                'ground_truth': processor.extract_answer(str(answer)),
                'accuracy': 1.0,  # Assume solutions are correct
                'dataset': domain_name
            })
    
    return processed
