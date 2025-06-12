# All code is original unless otherwise noted.
'''
python3 main.py \
  --train_json_file data/train.json \
  --meta_json_file  data/meta.json \
  --weights_path    outputs/math_prm \
  --batch_size      4 \
  --lr              1e-6 \
  --meta_lr         1e-2 \
  --iteration_num   5000 \
  --device          cuda

'''

import argparse
import torch.optim as optim 
from torch.optim import AdamW
from model import *
from data import *
from utils import *
from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig
import wandb
import numpy as np

# Argument parsing
default_pad = 151643
parser = argparse.ArgumentParser(description="DreamPRM Math Extension")
parser.add_argument('--train_json_file', type=str, required=True)
parser.add_argument('--meta_json_file', type=str, required=True)
parser.add_argument('--weights_path', type=str, required=True)
parser.add_argument('--iteration_num', type=int, default=10000)
parser.add_argument('--save_every_iterations', type=int, default=1000)
parser.add_argument('--unroll_steps', type=int, default=5)
parser.add_argument('--gradient_accumulation', type=int, default=1)
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--precision', type=str, default="bf16")
parser.add_argument('--strategy', type=str, default="default")
parser.add_argument('--rollback', action="store_true")
parser.add_argument('--baseline', action="store_true")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--lr', type=float, default=5e-7)
parser.add_argument('--meta_lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--meta_weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_epoch', type=int, default=120)
parser.add_argument('--meta_interval', type=int, default=1)
parser.add_argument('--unbalanced_factor', type=int, default=None)
parser.add_argument('--corruption_type', type=str, default=None)
parser.add_argument('--corruption_ratio', type=float, default=0.0)
args = parser.parse_args()
print(args)

# Reproducibility
set_seed(args.seed)
# Build domain list for weighting
domain_list = create_dataset_mapping(args.train_json_file)
print("Domains:", domain_list)

# DataLoaders for math PRM
train_loader, meta_loader = build_qwen_math_dataloader(
    tokenizer_path=args.model_name if hasattr(args, 'model_name') else "Qwen/Qwen2.5-Math-PRM-7B",
    train_json_file=args.train_json_file,
    meta_json_file=args.meta_json_file,
    train_batch_size=args.batch_size,
    meta_batch_size=args.batch_size
)

# Logging
device = torch.device(args.device)
wandb.init(project="DreamPRM_Math")

# Loss functions
criterion = torch.nn.MSELoss()
criterion_meta = torch.nn.MSELoss()

# Problem definitions
class Upper(ImplicitProblem):
    def forward(self, domain_strings, loss_tensor):
        # Domain scaling on lower loss
        return self.module(domain_strings, loss_tensor)

    def training_step(self, batch):
        # Meta step: evaluate aggregated rewards
        # batch: contains step inputs and labels
        step_keys = [k for k in batch.keys() if k.isdigit()]
        step_keys.sort(key=lambda x: int(x))
        # Gather step logits
        agg_logit = 0.0
        for key in step_keys:
            inp = batch[key]
            logits = self.inner(
                inp['input_ids'].to(device),
                inp['attention_mask'].to(device)
            )
            agg_logit += torch.log(logits / (1 - logits))
        # Normalize
        agg_prob = torch.sigmoid(agg_logit / len(step_keys))
        labels = batch['labels'].to(device)
        loss = criterion_meta(agg_prob, labels)
        return {"loss": loss}

    def configure_train_data_loader(self):
        return meta_loader

    def configure_module(self):
        return DomainTable(domain_list)

    def configure_optimizer(self):
        return AdamW(
            self.module.parameters(),
            lr=args.meta_lr,
            weight_decay=args.meta_weight_decay
        )

class Lower(ImplicitProblem):
    def forward(self, input_ids, attention_mask):
        return self.module(input_ids, attention_mask)

    def training_step(self, batch):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        domain_strings = batch['dataset']
        rewards = self.forward(input_ids, attention_mask)
        if args.baseline:
            loss = criterion(rewards, labels)
        else:
            # Weighted lower-level loss
            loss_raw = criterion(rewards, labels)
            loss = self.outer(domain_strings, loss_raw)
        return loss

    def configure_train_data_loader(self):
        return train_loader

    def configure_module(self):
        return QwenMath_RM(
            device=args.device,
            model_name="Qwen/Qwen2.5-Math-PRM-7B"
        )

    def configure_optimizer(self):
        return AdamW(
            self.module.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

# Engine configuration
upper_config = Config(
    type="darts",
    precision=args.precision,
    retain_graph=True
)
lower_config = Config(
    type="darts",
    precision=args.precision,
    unroll_steps=args.unroll_steps,
    gradient_accumulation=args.gradient_accumulation
)
engine_config = EngineConfig(
    train_iters=args.iteration_num,
    valid_step=args.save_every_iterations,
    strategy=args.strategy,
    roll_back=args.rollback,
    logger_type="wandb"
)

# Instantiate problems and engine
dep_dependencies = { 'l2u': {}, 'u2l': {} }
if not args.baseline:
    upper = Upper(name="upper", config=upper_config)
    lower = Lower(name="lower", config=lower_config)
    dep_dependencies['u2l'] = {upper: [lower]}
    dep_dependencies['l2u'] = {lower: [upper]}
    problems = [upper, lower]
else:
    lower = Lower(name="lower", config=lower_config)
    problems = [lower]

engine = Engine(
    config=engine_config,
    problems=problems,
    dependencies=dep_dependencies
)
engine.run()
