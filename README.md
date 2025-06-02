# DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning
## Table of Contents
1. [Method Overview](#installation)
2. [Quick Start](#quick-start)
3. [Customize Your Datasets](#customize-your-own-datasets)
4. [Citation](#citation)

## Method Overview <a name="installation"></a>

> **DreamPRM — Domain-Reweighted Process Reward Model for Multimodal Reasoning**  
> DreamPRM tackles the dataset *quality imbalance* and *distribution shift* that plague multimodal reasoning.  
> It jointly learns (i) a high-fidelity Process Reward Model (PRM) and (ii) optimal domain weights through a bi-level optimisation (BLO) loop, delivering a consistent **+4 pp** average gain on five public benchmarks. 

<!-- TODO: swap in your high-level diagram -->
![Training PRM and PRM for inference](figs/3.png)
![DreamPRM Overview](figs/5.png)


### Key Components

| Component | Purpose | Highlight |
|-----------|---------|-----------|
| **Domain-Reweighted Fine-Tuning** | Re-weights K training domains via parameters αₖ | Gives harder, higher-quality datasets greater gradient influence |
| **Bi-Level Optimisation (BLO)** | Lower level updates PRM weights ϕ; upper level updates α | Learns *both* model and data weights in one run |
| **Aggregation Function Loss** | Meta-level loss that mirrors inference-time scoring | Aligns training with real PRM usage |

<!-- TODO: swap in your domain-weight visualisation -->
![Learned domain weights](figs/6-4.png)

DreamPRM’s learned domain weights span **0.55–1.49**, down-weighting noisy sets like *AI2D* and up-weighting challenging ones like *M3CoT*. This correlation with dataset difficulty underpins its performance gains.

---

## Quick Start <a name="quick-start"></a>

> *All commands below are illustrative—rename scripts / paths to match your repo.*

### 1.  Environment

```bash
# (a) create conda env
conda create -n dreamprm python=3.10 -y
conda activate dreamprm

# (b) install requirements
pip install -r requirements.txt   # torch betty, transformers, accelerate, ...
```
### 2.  Domain-reweighting
Domain reweighting for PRM fine-tuning:
```bash
python reweighting/main.py --train_json_file "data/train.json" --meta_json_file "data/meta.json" --weights_path "weights"
```
## Customized Your Datasets <a name="customize-your-own-datasets"></a>
We provide demo datasets with 10 domains (10k training samples) and 500 meta samples in our repository:
```bash
data/
├── meta.json
└── train.json
```
### Training Dataset Format (for lower-level optimization)
Each sample in the training dataset should follow this format:
```python
{
    "id": 1128,                   # Unique question identifier
    "sid": 1,                     # Step number identifier
    "input": "Your task is...",    # Full question prompt
    "add": "Step 1: Restate...",   # Model's partial response
    "ground_truth": "1.78947",     # Correct final answer
    "image_path": "dataset/...",   # Path to input image
    "dataset": "chartqa",          # Domain name
    "score": 7,                    # Monte Carlo score
    "times": 11,                   # Monte Carlo iterations
    "accuracy": 0.6363             # Estimated accuracy (0-1)
}
```
### Minimal Custom Training Sample Format:
```python
{
    "input": "...",                # Question prompt (required)
    "add": "Step 1: ...",          # Model's partial response (required)
    "image_path": "xxx.png",       # Input image path (required)
    "dataset": "...",              # Domain name (required)
    "accuracy": 0.6363             # Estimated accuracy (0-1, required)
}
```
### Meta Dataset Format (for upper-level optimization)
Each sample in the meta dataset should follow this format:
```python
{
    "id": 2,                       # Unique question identifier
    "true_false": True,             # Ground truth label
    "input": "Question: The...",    # Full question + model response
    "image_path": "dataset/..."     # Path to input image
}
```
### Minimal Custom Meta Sample Format:
```python
{
    "true_false": True,             # Boolean ground truth (required)
    "input": "Question: ...",       # Full question + model response (required)
    "image_path": "xxx.png"         # Input image path (required)
}
```

## Citation <a name="citation"></a>
If you use this work in your research, please cite:
```bibtex
@misc{cao2025dreamprmdomainreweightedprocessreward,
      title={DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning}, 
      author={Qi Cao and Ruiyi Wang and Ruiyi Zhang and Sai Ashish Somayajula and Pengtao Xie},
      year={2025},
      eprint={2505.20241},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.20241}, 
}
```