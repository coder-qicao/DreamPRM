# 医学影像分析适配总结

## 完成的工作

我已经为你成功适配了 segmentation.json 到 DreamPRM 仓库，创建了完整的医学影像三分类任务流程。

### 1. 创建的核心脚本

| 文件 | 功能 | 说明 |
|------|------|------|
| `generate_cot_segmentation.py` | CoT 生成 | 使用 GPT-4o/o4-mini API 生成结构化思维链，支持生成 8 个不同的 CoT 用于 best-of-N 选择 |
| `preprocess_segmentation.py` | 数据预处理 | 验证数据完整性，选择有效 PNG 图片，生成数据统计 |
| `evaluate_gpt.py` | 评估 | 支持三种评估模式：zero-shot、best-of-N、PRM-guided |
| `main_medical.py` | PRM 训练 | 适配的训练脚本，使用双层优化学习 domain weights |

### 2. 辅助文件

| 文件 | 功能 |
|------|------|
| `README_MEDICAL.md` | 详细使用文档，包含完整的工作流程和参数说明 |
| `config_medical.yaml` | 配置文件模板，集中管理所有参数 |
| `run_medical_pipeline.sh` | 一键运行完整流程的 Bash 脚本 |
| `test_medical_setup.py` | 环境检查脚本，验证所有依赖和文件 |
| `SUMMARY.md` | 本文档 |

## 任务理解

### DreamPRM 核心原理

1. **训练阶段**：
   - **Lower-level**: 训练 Process Reward Model (PRM)，预测每个 CoT 步骤的准确率
   - **Upper-level**: 学习 domain weights，对不同数据集/任务进行重新加权
   - 通过双层优化 (Bi-level Optimization) 同时优化模型和数据权重

2. **推理阶段**：
   - 生成多个 CoT (Chain-of-Thought)
   - 使用 PRM 对每个 CoT 的步骤进行评分
   - 选择总分最高的 CoT 作为最终答案

### 医学影像任务适配

**任务**：预测宫颈癌 MRI 扫描的三个二分类标签
- Lymphatic_node_transfer (淋巴结转移)
- Invade the uterus (侵犯子宫)
- Invade the vagina (侵犯阴道)

**数据流程**：
```
segmentation.json
  → 选择有效 PNG 图片
  → 调用 GPT API 生成结构化 CoT（5 步推理）
  → 提取步骤级数据（训练数据）
  → 训练 PRM 或直接评估
```

**CoT 结构**（5 步）：
1. Restate the question（重述问题）
2. Gather evidence（收集证据）
3. Identify background knowledge（识别背景知识）
4. Reason using evidence（基于证据推理）
5. Summarize and conclude（总结并得出结论）

## 数据格式说明

### 输入数据 (segmentation.json)

```json
{
  "patient": "sfybl_0",
  "Image description": "图像描述...",
  "image diagnosis": "影像诊断...",
  "Lymphatic_node_transfer": "No",
  "Invade the uterus": "No",
  "Invade the vagina": "No",
  "segmentation_images": ["/path/to/image.png"],
  "age": "58"
}
```

### 生成的训练数据 (medical_train.json)

```json
{
  "id": "sfybl_0_Lymphatic_node_transfer_0_1",
  "sid": 1,
  "input": "完整结构化提示...",
  "add": "Step 1: Restate the question...",
  "ground_truth": "no",
  "image_path": "/path/to/image.png",
  "dataset": "medical_segmentation",
  "task": "Lymphatic_node_transfer",
  "accuracy": 0.95,
  "is_correct": true
}
```

### 生成的 Meta 数据 (medical_meta.json)

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

## 快速开始

### 1. 环境检查

```bash
python test_medical_setup.py
```

### 2. 运行完整流程

```bash
# 一键运行（推荐）
bash run_medical_pipeline.sh

# 或者分步运行
python preprocess_segmentation.py --segmentation_json segmentation.json --select_valid_images
python generate_cot_segmentation.py --segmentation_json segmentation.json --max_patients 10
python evaluate_gpt.py --meta_json data/medical_meta.json --eval_mode best_of_n
```

### 3. 查看结果

```bash
# 查看评估结果
cat results_best_of_n.json | jq

# 对比 zero-shot 和 best-of-N
cat results_zero_shot.json | jq
```

## 核心特性

### 1. CoT 生成

- **模型选择**: 支持 o4-mini（高推理能力）和 gpt-4o（速度快）
- **推理力度**: o4-mini 支持 low/medium/high 三种推理强度
- **Best-of-N**: 每个问题生成 8 个不同的 CoT，提高答案质量
- **结构化提示**: 使用 5 步结构化提示确保推理质量

### 2. 评估模式

- **Zero-shot**: 直接预测，作为基线
- **Best-of-N**: 从多个 CoT 中通过投票选择最佳答案
- **PRM-guided**: 使用训练好的 PRM 对 CoT 评分并选择

### 3. Domain Reweighting

- 通过双层优化学习每个任务的权重
- 自动识别难度较高的任务并增加权重
- 提高整体模型性能

## 关键优势

1. **准确率提升**: Best-of-N 相比 zero-shot 通常有显著提升
2. **可解释性**: 结构化的 CoT 提供了清晰的推理过程
3. **灵活性**: 支持多种评估模式和模型选择
4. **可扩展性**: 易于适配其他医学影像任务

## 下一步建议

### 短期（立即可做）

1. **运行测试**:
   ```bash
   python test_medical_setup.py
   bash run_medical_pipeline.sh
   ```

2. **分析结果**: 对比不同评估模式的性能

3. **调整参数**:
   - 修改 `config_medical.yaml` 中的参数
   - 尝试不同的 CoT 数量和推理强度

### 中期（1-2 周）

1. **数据扩展**: 处理更多患者数据（去除 `--max_patients` 限制）

2. **训练 PRM**:
   ```bash
   python main_medical.py \
     --train_json_file data/medical_train.json \
     --meta_json_file data/medical_meta.json \
     --weights_path weights_medical \
     --iteration_num 10000
   ```

3. **超参数调优**: 调整学习率、domain weights 等参数

### 长期（1-3 个月）

1. **集成到生产环境**:
   - 创建 API 服务
   - 添加批处理支持
   - 实现缓存机制

2. **多任务扩展**: 适配其他医学影像任务

3. **性能优化**:
   - 并行处理
   - 图片压缩
   - API 调用优化

## 技术要点

### CoT 生成原理

根据 README.md 中的说明，o4-mini 使用高推理强度模式：

```python
client.responses.create(
    model="o4-mini",
    reasoning={"effort": "high"},
    input=structured_prompt
)
```

建议每个问题生成 **8 个不同的 CoT** 以实现 best-of-N 选择。

### Best-of-N 选择

从多个 CoT 中选择最佳答案的方法：

1. **简单投票**: 选择出现次数最多的答案
2. **加权投票**: 根据 CoT 质量（步骤完整性、置信度）加权
3. **PRM 评分**: 使用训练好的 PRM 计算每个 CoT 的分数

### Domain Weighting

DreamPRM 学习到的 domain weights 范围通常在 **0.55-1.49**：
- 低权重（~0.55）: 数据质量高、任务简单
- 高权重（~1.49）: 数据噪声多、任务困难

这种自适应加权能够提升整体性能。

## 故障排除

### 常见问题

1. **API 调用失败 (429 错误)**
   - 原因: 速率限制
   - 解决: 脚本已实现指数退避重试，如仍失败请增加重试间隔

2. **图片路径错误**
   - 原因: segmentation.json 中的路径无效
   - 解决: 运行 `preprocess_segmentation.py --check_images` 验证

3. **内存不足**
   - 原因: 处理大量图片
   - 解决: 使用 `--max_patients` 限制数量，或分批处理

4. **CoT 格式不一致**
   - 原因: GPT 返回的格式可能变化
   - 解决: 检查提取步骤的正则表达式，必要时调整

## 文件清单

```
DreamPRM/
├── generate_cot_segmentation.py     # CoT 生成脚本
├── preprocess_segmentation.py       # 数据预处理
├── evaluate_gpt.py                  # 评估脚本
├── main_medical.py                  # PRM 训练
├── test_medical_setup.py            # 环境检查
├── run_medical_pipeline.sh          # 完整流程脚本
├── config_medical.yaml              # 配置文件
├── README_MEDICAL.md                # 详细文档
├── SUMMARY.md                       # 本文档
├── api.txt                          # API key（需创建）
├── segmentation.json                # 输入数据（需提供）
└── data/
    ├── medical_train.json           # 训练数据（生成）
    └── medical_meta.json            # Meta 数据（生成）
```

## 联系和反馈

如有问题或建议，请：
1. 查看 `README_MEDICAL.md` 详细文档
2. 运行 `python test_medical_setup.py` 检查环境
3. 检查生成的日志和错误信息

## 参考资料

- **DreamPRM 论文**: https://arxiv.org/abs/2505.20241
- **原始 README**: README.md
- **医学任务文档**: README_MEDICAL.md
- **Template 参考**: template.md
- **Zero-shot 示例**: zero_shot_base_template.py

---

**创建日期**: 2025-12-16
**版本**: 1.0
**作者**: Claude Code
