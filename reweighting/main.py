# Written by QI CAO on May 21, 2025.
# All code is original unless otherwise noted.

import sys
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='Project root path')
parser.add_argument('--gpu', type=str, default='0', help='GPU device ID (CUDA_VISIBLE_DEVICES)')
args = parser.parse_args()

# Append project root path to sys.path for module importing
sys.path.append(args.path)

# Set visible CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Change working directory to the project root
os.chdir(args.path)

# Optional: Print confirmation
print(f"CUDA_VISIBLE_DEVICES set to {args.gpu}")
print(f"Working directory changed to {args.path}")

# main
import torch
import torch.optim as optim
from reweighting.model import *
from reweighting.data import *
from reweighting.utils import *
from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig
import wandb
from transformers import AutoProcessor, AdamW
import numpy as np


parser = argparse.ArgumentParser(description="Meta_Weight_Net")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--precision", type=str, default="bf16")
parser.add_argument("--strategy", type=str, default="default")
parser.add_argument("--rollback", action="store_true")
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--retrain", action="store_true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--meta_net_hidden_size", type=int, default=100)
parser.add_argument("--meta_net_num_layers", type=int, default=1)
parser.add_argument("--lr", type=float, default=5e-7)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--dampening", type=float, default=0.0)
parser.add_argument("--nesterov", type=bool, default=False)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--meta_lr", type=float, default=0.01)
parser.add_argument("--meta_weight_decay", type=float, default=0.0)
parser.add_argument("--dataset", type=str, default="cifar10")
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

sampler = None
resume_idxes = None
resume_labels = None
# if args.retrain:
    # sample_weight = torch.load("reweight.pt")
    # resume_idxes = torch.load("train_index.pt")
    # resume_labels = torch.load("train_label.pt")
    # sampler = WeightedRandomSampler(sample_weight, len(sample_weight))

(
    train_dataloader,
    meta_dataloader,
    test_dataloader,
) = build_dataloader(
    processor_path = "Qwen/Qwen2-VL-2B-Instruct",
    train_json_file = "reweighting/MMPR/InternVL-MPO/train.json",
    meta_json_file = "reweighting/MMPR/InternVL-MPO/meta.json",
    test_json_file = "reweighting/MMPR/InternVL-MPO/train.json",
    train_batch_size= 1,
    meta_batch_size= 1,
)
wandb.init(project="PRM_bi_level")
# print(Counter(train_dataloader.dataset.targets))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()
criterion_meta = nn.MSELoss()
inner_weighted_loss = []
inner_loss = []
outer_loss = []
best_loss = 1000


class Outer(ImplicitProblem):
    def forward(self, domain_strings, x):
        # torch.cuda.empty_cache()
        return self.module(domain_strings, x)

    def training_step(self, batch):
        steps = [batch['1'], batch['2'], batch['3'], batch['4'], batch['5'],]
        labels = batch['labels'].to(device)
        mean_score = 0
        for i in steps:
            score = self.inner(i['input_ids'].to(device),
                                     i['attention_mask'].to(device),
                                     i['pixel_values'].to(device),
                                     i['image_grid_thw'].to(device))
            mean_score += torch.log(score / (1 - score))
        outputs = torch.sigmoid(mean_score / 5)
        loss = criterion_meta(outputs, labels)
        outer_loss.append(loss.item())
        positive_weights = torch.nn.functional.softplus(self.module.raw_weights)
        mean_weights = positive_weights.mean()
        print(outputs.item(), labels.item(), loss.item())
        normalized_weights = positive_weights / mean_weights
        wandb.log({
               "ai2d_train_12k_en_20240410_extracted_pairs_vqa_correctness_rules":normalized_weights[0].item(),
               "chartqa_trainval_30k_w_csv_en_20240402_extracted_pairs_vqa_correctness_rules":normalized_weights[1].item(),
               "m3cot_train_extracted_pairs_vqa_correctness_rules":normalized_weights[2].item(),
               "scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_correctness_rules":normalized_weights[3].item(),
               "mapqa_suv_en_20240402_extracted_pairs_vqa_correctness_rules":normalized_weights[4].item(),
               "geo170k_extracted_full_pairs_vqa_correctness_rules":normalized_weights[5].item(),
               "CLEVR_math_en_20240402_extracted_pairs_vqa_correctness_rules":normalized_weights[6].item(),
               "geometry3k_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules":normalized_weights[7].item(),
               "figureqa_en_20240402_extracted_pairs_vqa_correctness_rules":normalized_weights[8].item(),
               "infographics_20240403_qa_20240407_v2_extracted_pairs_vqa_correctness_rules":normalized_weights[9].item(),
               "unigeo_calc_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules":normalized_weights[10].item(),
               "geomverse_extracted_pairs_vqa_correctness_rules":normalized_weights[11].item(),
               "iconqa_train_extracted_pairs_vqa_correctness_rules":normalized_weights[12].item(),
               "dvqa_en_20240402_extracted_int_only_pairs_vqa_correctness_rules":normalized_weights[13].item(),
               "geos_en_20240402_extracted_pairs_vqa_correctness_rules":normalized_weights[14].item(),
        })
        # torch.cuda.empty_cache()
        if len(outer_loss) == 10:
            mean_outer_loss = np.mean(outer_loss)
            wandb.log({"outer_loss": mean_outer_loss})
            outer_loss.clear()

        return {"loss": loss}

    def configure_train_data_loader(self):
        return meta_dataloader

    def configure_module(self):
        meta_net = DomainTable(
            {"ai2d_train_12k_en_20240410_extracted_pairs_vqa_correctness_rules":0,
               "chartqa_trainval_30k_w_csv_en_20240402_extracted_pairs_vqa_correctness_rules":1,
               "m3cot_train_extracted_pairs_vqa_correctness_rules":2,
               "scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_correctness_rules":3,
               "mapqa_suv_en_20240402_extracted_pairs_vqa_correctness_rules":4,
               "geo170k_extracted_full_pairs_vqa_correctness_rules":5,
               "CLEVR_math_en_20240402_extracted_pairs_vqa_correctness_rules":6,
               "geometry3k_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules":7,
               "figureqa_en_20240402_extracted_pairs_vqa_correctness_rules":8,
               "infographics_20240403_qa_20240407_v2_extracted_pairs_vqa_correctness_rules":9,
               "unigeo_calc_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules":10,
               "geomverse_extracted_pairs_vqa_correctness_rules":11,
               "iconqa_train_extracted_pairs_vqa_correctness_rules":12,
               "dvqa_en_20240402_extracted_int_only_pairs_vqa_correctness_rules":13,
               "geos_en_20240402_extracted_pairs_vqa_correctness_rules":14,
             }
        )
        return meta_net

    def configure_optimizer(self):
        meta_optimizer = AdamW(
            self.module.parameters(),
            lr=args.meta_lr,
            weight_decay=args.weight_decay
        )
        return meta_optimizer


class Inner(ImplicitProblem):
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
        weighted_loss = self.outer(domain_strings, loss)
        inner_loss.append(loss.item())
        inner_weighted_loss.append(weighted_loss.item())
        if len(inner_loss) == 100:
            mean_inner_loss = np.mean(inner_loss)
            mean_inner_weighted_loss = np.mean(inner_weighted_loss)
            wandb.log({"inner_loss": mean_inner_loss,
                       "inner_weighted_loss": mean_inner_weighted_loss, })
            inner_loss.clear()
            inner_weighted_loss.clear()
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
            self.optimizer, step_size = 5000, gamma=0.5
        )
        return scheduler



class ReweightingEngine(Engine):
    @torch.no_grad()
    def validation(self):
        torch.save(
            self.inner.module.LN.state_dict(), f"reweighting/weights_3/LN_weights.pt"
        )
        self.inner.module.base_model.save_pretrained("reweighting/weights_3/base_model")
        torch.save(
            self.outer.state_dict(),
            f"reweighting/weights_3/domain_weights.pt",
        )
        return {"loss": 1}


outer_config = Config(type="darts", precision=args.precision, retain_graph=True)
inner_config = Config(type="darts", precision=args.precision, unroll_steps=5, gradient_accumulation=1)
engine_config = EngineConfig(
    train_iters=10000,
    valid_step=1000,
    strategy=args.strategy,
    roll_back=args.rollback,
    logger_type="wandb",
)
outer = Outer(name="outer", config=outer_config)
inner = Inner(name="inner", config=inner_config)

if args.baseline or args.retrain:
    problems = [inner]
    u2l, l2u = {}, {}
else:
    problems = [outer, inner]
    u2l = {outer: [inner]}
    l2u = {inner: [outer]}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = ReweightingEngine(
    config=engine_config, problems=problems, dependencies=dependencies
)
engine.run()
