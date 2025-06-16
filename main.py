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


parser = argparse.ArgumentParser(description="DreamPRM")
parser.add_argument('--train_json_file', type=str)
parser.add_argument('--meta_json_file', type=str)
parser.add_argument('--weights_path', type=str)
parser.add_argument("--iteration_num", type=int, default=10000)
parser.add_argument("--save_every_iterations", type=int, default=1000)
parser.add_argument("--unroll_steps", type=int, default=5)
parser.add_argument("--gradiant_accumulation", type=int, default=1)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--precision", type=str, default="bf16")
parser.add_argument("--strategy", type=str, default="default")
parser.add_argument("--rollback", action="store_true")
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--lr", type=float, default=5e-7)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--scheduler_step_size", type=int, default=5000)
parser.add_argument("--scheduler_gamma", type=float, default=0.5)
parser.add_argument("--dampening", type=float, default=0.0)
parser.add_argument("--nesterov", type=bool, default=False)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--meta_lr", type=float, default=0.01)
parser.add_argument("--meta_weight_decay", type=float, default=0.0)
parser.add_argument("--reward_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
parser.add_argument("--num_meta", type=int, default=1000)
parser.add_argument("--imbalanced_factor", type=int, default=None)
parser.add_argument("--corruption_type", type=str, default=None)
parser.add_argument("--corruption_ratio", type=float, default=0.0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_epoch", type=int, default=120)
parser.add_argument("--meta_interval", type=int, default=1)
parser.add_argument("--paint_interval", type=int, default=20)

args = parser.parse_args()
print(args)
set_seed(args.seed)
domain_list = create_dataset_mapping(args.train_json_file)
print(domain_list)

sampler = None
resume_idxes = None
resume_labels = None

(
    train_dataloader,
    meta_dataloader,
) = build_dataloader(
    processor_path = args.reward_model,
    train_json_file = args.train_json_file,
    meta_json_file = args.meta_json_file,
    train_batch_size= args.batch_size,
    meta_batch_size= args.batch_size,
)
wandb.init(project="DreamPRM")

device = torch.device(args.device)
criterion = nn.MSELoss()
criterion_meta = nn.MSELoss()
lower_weighted_loss = []
lower_loss = []
upper_loss = []
best_loss = 1000


class Upper(ImplicitProblem):
    def forward(self, domain_strings, x):
        # torch.cuda.empty_cache()
        return self.module(domain_strings, x)

    def training_step(self, batch):
        # steps = [batch['1'], batch['2'], batch['3'], batch['4'], batch['5'],]
        numeric_keys = [k for k in batch.keys() if k.isdigit()]
        sorted_keys = sorted(numeric_keys, key=lambda x: int(x))
        steps = [batch[key] for key in sorted_keys]
        labels = batch['labels'].to(device)
        mean_score = 0
        for i in steps:
            score = self.inner(i['input_ids'].to(device),
                                     i['attention_mask'].to(device),
                                     i['pixel_values'].to(device),
                                     i['image_grid_thw'].to(device))
            mean_score += torch.log(score / (1 - score))
        outputs = torch.sigmoid(mean_score / len(steps))
        loss = criterion_meta(outputs, labels)
        upper_loss.append(loss.item())
        print(outputs.item(), labels.item(), loss.item())
        # torch.cuda.empty_cache()
        if len(upper_loss) == 10:
            mean_outer_loss = np.mean(upper_loss)
            wandb.log({"outer_loss": mean_outer_loss})
            upper_loss.clear()

        return {"loss": loss}

    def configure_train_data_loader(self):
        return meta_dataloader

    def configure_module(self):
        meta_net = DomainTable(
            domain_list
        )
        return meta_net

    def configure_optimizer(self):
        meta_optimizer = AdamW(
            self.module.parameters(),
            lr=args.meta_lr,
            weight_decay=args.weight_decay
        )
        return meta_optimizer


class Lower(ImplicitProblem):
    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        # torch.cuda.empty_cache()
        return self.module(input_ids, attention_mask, pixel_values, image_grid_thw)

    def training_step(self, batch):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        image_grid_thw = batch['image_grid_thw'].to(device)
        labels = batch['label'].to(dtype=torch.float).to(device)
        domain_strings = batch['dataset']
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
                     image_grid_thw=image_grid_thw)
        if args.baseline or args.retrain:
            return criterion(outputs, labels)
        loss = criterion(outputs, labels)
        weighted_loss = self.upper(domain_strings, loss)
        lower_loss.append(loss.item())
        lower_weighted_loss.append(weighted_loss.item())
        if len(lower_loss) == 100:
            mean_inner_loss = np.mean(lower_loss)
            mean_inner_weighted_loss = np.mean(lower_weighted_loss)
            wandb.log({"inner_loss": mean_inner_loss,
                       "inner_weighted_loss": mean_inner_weighted_loss, })
            lower_loss.clear()
            lower_weighted_loss.clear()
        # torch.cuda.empty_cache()

        return weighted_loss

    def configure_train_data_loader(self):
        return train_dataloader

    def configure_module(self):
        return QwenVL_RM(device)

    def configure_optimizer(self):
        optimizer = AdamW(
            self.module.parameters(),
            lr=args.lr,
        )
        return optimizer

    def configure_scheduler(self):
        scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size = args.scheduler_step_size, gamma=args.scheduler_gamma
        )
        return scheduler



class ReweightingEngine(Engine):
    @torch.no_grad()
    def validation(self):
        torch.save(
            self.inner.module.LN.state_dict(), f"{args.weights_path}/LN_weights.pt"
        )
        self.inner.module.base_model.save_pretrained(f"{args.weights_path}/base_model")
        torch.save(
            self.outer.state_dict(),
            f"{args.weights_path}/domain_weights.pt",
        )
        return {"loss": 1}


upper_config = Config(type="darts", precision=args.precision, retain_graph=True)
lower_config = Config(type="darts", precision=args.precision, unroll_steps=args.unroll_steps, gradient_accumulation=args.gradiant_accumulation)
engine_config = EngineConfig(
    train_iters=args.iteration_num,
    valid_step=args.parser.save_every_iterations,
    strategy=args.strategy,
    roll_back=args.rollback,
    logger_type="wandb",
)
upper = Upper(name="upper", config=upper_config)
lower = Lower(name="lower", config=lower_config)

if args.baseline or args.retrain:
    problems = [lower]
    u2l, l2u = {}, {}
else:
    problems = [upper, lower]
    u2l = {upper: [lower]}
    l2u = {lower: [upper]}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = ReweightingEngine(
    config=engine_config, problems=problems, dependencies=dependencies
)
engine.run()