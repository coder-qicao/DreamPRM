# 医学影像分析适配 - DreamPRM

本文档说明如何将 segmentation.json 数据适配到 DreamPRM 框架，用于医学影像的三分类任务。

## 任务概述

**目标**: 预测宫颈癌 MRI 扫描的三个二分类标签：
1. **Lymphatic_node_transfer** (淋巴结转移): yes/no
2. **Invade the uterus** (侵犯子宫): yes/no
3. **Invade the vagina** (侵犯阴道): yes/no

**数据**: segmentation.json 包含患者的医学影像数据和文本描述

**方法**: 使用 GPT-4o/o4-mini API 生成结构化的 Chain-of-Thought (CoT)，然后训练 Process Reward Model (PRM) 进行评分和选择

## 工作流程

```
segmentation.json
       ↓
[1. 数据预处理] → segmentation_processed.json
       ↓
[2. CoT 生成] → medical_train.json + medical_meta.json
       ↓
[3. PRM 训练] → trained PRM weights
       ↓
[4. 评估] → evaluation_results.json
```

## 快速开始

### 前提条件

1. 安装依赖:
```bash
pip install -r requirements.txt
```

2. 准备 API Key:
```bash
echo "your-api-key-here" > api.txt
```

3. 准备数据:
- 将 `segmentation.json` 放在项目根目录
- 确保图片路径可访问

### 步骤 1: 数据预处理

验证 segmentation.json 数据的完整性和有效性:

```bash
python preprocess_segmentation.py \
  --segmentation_json segmentation.json \
  --output_json segmentation_processed.json \
  --select_valid_images \
  --check_images
```

**参数说明:**
- `--segmentation_json`: 输入的 segmentation.json 文件路径
- `--output_json`: 输出的处理后 JSON 文件
- `--select_valid_images`: 自动选择有效的 PNG 图片
- `--check_images`: 检查所有图片是否存在且有效

**输出:**
- `segmentation_processed.json`: 处理后的数据
- 终端输出数据统计信息（患者数、标签分布等）

### 步骤 2: 生成 CoT 数据

使用 GPT-4o/o4-mini API 生成结构化的思维链推理:

```bash
python generate_cot_segmentation.py \
  --api_key_path api.txt \
  --segmentation_json segmentation.json \
  --output_train_json data/medical_train.json \
  --output_meta_json data/medical_meta.json \
  --model o4-mini \
  --num_cots 8 \
  --reasoning_effort high \
  --max_patients 10
```

**参数说明:**
- `--model`: 使用的模型 (`o4-mini` 或 `gpt-4o`)
  - `o4-mini`: 推理能力强，支持高强度推理模式
  - `gpt-4o`: 标准视觉模型，速度快
- `--num_cots`: 每个问题生成的 CoT 数量（建议 8 个用于 best-of-N）
- `--reasoning_effort`: o4-mini 的推理强度 (`low`, `medium`, `high`)
- `--max_patients`: 限制处理的患者数量（用于测试，None 表示全部）

**输出:**
- `data/medical_train.json`: 训练数据（包含步骤级样本）
- `data/medical_meta.json`: Meta 数据（包含完整 CoT）

**数据格式:**

训练数据样本:
```json
{
  "id": "sfybl_0_Lymphatic_node_transfer_0_1",
  "sid": 1,
  "input": "完整的结构化提示...",
  "add": "Step 1: Restate the question...",
  "ground_truth": "no",
  "image_path": "/path/to/image.png",
  "dataset": "medical_segmentation",
  "task": "Lymphatic_node_transfer",
  "accuracy": 0.95,
  "is_correct": true
}
```

Meta 数据样本:
```json
{
  "id": "sfybl_0_Lymphatic_node_transfer_0",
  "input": "完整提示 + 完整 CoT...",
  "image_path": "/path/to/image.png",
  "true_false": true,
  "task": "Lymphatic_node_transfer",
  "ground_truth": "no",
  "predicted_answer": "no"
}
```

### 步骤 3: 训练 PRM（可选）

如果你想训练自己的 Process Reward Model:

```bash
python main_medical.py \
  --train_json_file data/medical_train.json \
  --meta_json_file data/medical_meta.json \
  --weights_path weights_medical \
  --iteration_num 5000 \
  --lr 1e-4 \
  --meta_lr 0.01 \
  --use_wandb
```

**参数说明:**
- `--iteration_num`: 训练迭代次数
- `--lr`: PRM 学习率
- `--meta_lr`: Domain weights 学习率
- `--use_wandb`: 使用 Weights & Biases 记录训练过程

**输出:**
- `weights_medical/prm_weights.pt`: PRM 模型权重
- `weights_medical/domain_weights.pt`: Domain weights

### 步骤 4: 评估

#### 4.1 Zero-shot 评估（无 CoT）

直接使用 GPT 进行预测，不使用 CoT:

```bash
python evaluate_gpt.py \
  --api_key_path api.txt \
  --segmentation_json segmentation.json \
  --eval_mode zero_shot \
  --model gpt-4o \
  --output_results results_zero_shot.json \
  --max_patients 10
```

#### 4.2 Best-of-N 评估

从多个 CoT 中通过投票选择最佳答案:

```bash
python evaluate_gpt.py \
  --api_key_path api.txt \
  --meta_json data/medical_meta.json \
  --eval_mode best_of_n \
  --output_results results_best_of_n.json
```

#### 4.3 PRM-guided 评估（需要训练好的 PRM）

使用训练好的 PRM 对 CoT 进行评分并选择最佳:

```bash
python evaluate_gpt.py \
  --api_key_path api.txt \
  --meta_json data/medical_meta.json \
  --eval_mode prm_guided \
  --output_results results_prm_guided.json
```

**评估输出:**
```
=== 评估结果汇总 ===
总样本数: 30
正确预测: 24
总体准确率: 80.00%

按任务统计:
  Lymphatic_node_transfer: 8/10 = 80.00%
  Invade the uterus: 9/10 = 90.00%
  Invade the vagina: 7/10 = 70.00%
```

## 详细说明

### CoT 结构化提示

生成的 CoT 遵循以下 5 步结构:

```
Step 1: Restate the question
  - 重新阐述分类任务

Step 2: Gather evidence from image and text
  - 从图像和文本描述中收集证据
  - 关注：病灶特征、解剖结构、信号模式、侵犯范围

Step 3: Identify medical background knowledge
  - 列出相关的医学知识（宫颈癌分期、侵犯模式）

Step 4: Reason using evidence
  - 整合图像发现、文本描述和医学知识进行推理
  - 考虑：肿瘤大小、位置、信号特征、与邻近结构的关系

Step 5: Summarize and conclude
  - 提供明确的分类结果（yes/no），并给出支持理由

Final answer: [yes/no]
```

### Domain Weighting

DreamPRM 通过双层优化学习 domain weights:

- **Lower-level**: 训练 PRM 预测每个 CoT 步骤的准确率
- **Upper-level**: 学习 domain weights，对不同数据集/任务进行重新加权

学习到的 domain weights 反映了不同任务的难度和质量。

### Best-of-N 选择策略

当生成多个 CoT 时，可以使用以下策略选择最佳答案:

1. **简单投票** (Majority Voting): 选择出现次数最多的答案
2. **加权投票**: 根据 CoT 质量（例如步骤完整性）进行加权
3. **PRM 评分**: 使用训练好的 PRM 对每个 CoT 评分，选择分数最高的

## 文件说明

| 文件 | 说明 |
|------|------|
| `generate_cot_segmentation.py` | CoT 生成脚本（调用 GPT API） |
| `preprocess_segmentation.py` | 数据预处理脚本 |
| `evaluate_gpt.py` | 评估脚本（支持三种模式） |
| `main_medical.py` | 医学影像任务的 PRM 训练脚本 |
| `README_MEDICAL.md` | 本说明文档 |

## 数据要求

### 输入数据格式 (segmentation.json)

```json
[
  {
    "patient": "sfybl_0",
    "Image description": "图像描述文本...",
    "image diagnosis": "影像诊断文本...",
    "Lymphatic_node_transfer": "No",
    "Invade the uterus": "No",
    "Invade the vagina": "No",
    "segmentation_images": [
      "/path/to/image_0.png",
      "/path/to/image_1.png"
    ],
    "age": "58"
  }
]
```

**必需字段:**
- `patient`: 患者 ID
- `Image description`: 图像描述
- `image diagnosis`: 影像诊断
- `Lymphatic_node_transfer`: 淋巴结转移标签 (yes/no)
- `Invade the uterus`: 侵犯子宫标签 (yes/no)
- `Invade the vagina`: 侵犯阴道标签 (yes/no)
- `segmentation_images`: 图片路径列表

**可选字段:**
- `age`: 患者年龄
- 其他医学信息字段

## 性能优化建议

1. **批处理**: 如果有大量患者数据，可以分批处理以避免 API 限流
2. **缓存**: 对已生成的 CoT 进行缓存，避免重复调用 API
3. **并行处理**: 使用多线程/多进程并行生成 CoT（注意 API 速率限制）
4. **图片压缩**: 对大图片进行适当压缩以减少 API 调用时间

## 故障排除

### API 调用失败

**问题**: `API 调用失败: 429 Too Many Requests`

**解决方案**:
- 在脚本中增加重试间隔（已实现指数退避）
- 减少并发请求数量
- 联系 API 提供商增加速率限制

### 图片路径错误

**问题**: `警告: 图片不存在 - /path/to/image.png`

**解决方案**:
- 检查 segmentation.json 中的图片路径是否正确
- 确保图片文件存在且可读
- 使用 `preprocess_segmentation.py --check_images` 验证所有图片

### 内存不足

**问题**: 处理大量图片时内存不足

**解决方案**:
- 使用 `--max_patients` 参数限制处理数量
- 分批处理患者数据
- 增加系统内存或使用更小的图片分辨率

## 引用

如果你在研究中使用了 DreamPRM，请引用:

```bibtex
@misc{cao2025dreamprmdomainreweightedprocessreward,
      title={DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning},
      author={Qi Cao and Ruiyi Wang and Ruiyi Zhang and Sai Ashish Somayajula and Pengtao Xie},
      year={2025},
      eprint={2505.20241},
      archivePrefix={arXiv}
}
```

## 许可证

本项目遵循 Apache License 2.0。详见 [LICENSE.md](LICENSE.md)。
