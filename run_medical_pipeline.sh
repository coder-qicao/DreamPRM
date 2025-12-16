#!/bin/bash
# 医学影像分析完整流程脚本
# 用法: bash run_medical_pipeline.sh

set -e  # 遇到错误立即退出

echo "========================================="
echo "医学影像分析 - DreamPRM 完整流程"
echo "========================================="
echo ""

# ===== 配置参数 =====
SEGMENTATION_JSON="segmentation.json"
API_KEY_PATH="api.txt"
MAX_PATIENTS=10  # 测试时限制患者数量，设置为 0 表示处理全部
NUM_COTS=8
MODEL="o4-mini"
REASONING_EFFORT="high"

# ===== 检查必要文件 =====
echo "步骤 0: 检查必要文件..."

if [ ! -f "$SEGMENTATION_JSON" ]; then
    echo "错误: 找不到 $SEGMENTATION_JSON"
    echo "请将 segmentation.json 放在项目根目录"
    exit 1
fi

if [ ! -f "$API_KEY_PATH" ]; then
    echo "错误: 找不到 $API_KEY_PATH"
    echo "请创建 api.txt 并写入你的 API key"
    exit 1
fi

echo "✓ 文件检查完成"
echo ""

# ===== 步骤 1: 数据预处理 =====
echo "步骤 1: 数据预处理..."
echo "----------------------------------------"

python preprocess_segmentation.py \
  --segmentation_json "$SEGMENTATION_JSON" \
  --output_json segmentation_processed.json \
  --select_valid_images \
  --check_images

echo ""
echo "✓ 数据预处理完成"
echo ""

# ===== 步骤 2: 生成 CoT 数据 =====
echo "步骤 2: 生成 CoT 数据..."
echo "----------------------------------------"
echo "使用模型: $MODEL"
echo "每个问题生成 $NUM_COTS 个 CoT"
if [ $MAX_PATIENTS -gt 0 ]; then
    echo "限制处理前 $MAX_PATIENTS 个患者（测试模式）"
fi
echo ""

if [ $MAX_PATIENTS -gt 0 ]; then
    python generate_cot_segmentation.py \
      --api_key_path "$API_KEY_PATH" \
      --segmentation_json "$SEGMENTATION_JSON" \
      --output_train_json data/medical_train.json \
      --output_meta_json data/medical_meta.json \
      --model "$MODEL" \
      --num_cots $NUM_COTS \
      --reasoning_effort "$REASONING_EFFORT" \
      --max_patients $MAX_PATIENTS
else
    python generate_cot_segmentation.py \
      --api_key_path "$API_KEY_PATH" \
      --segmentation_json "$SEGMENTATION_JSON" \
      --output_train_json data/medical_train.json \
      --output_meta_json data/medical_meta.json \
      --model "$MODEL" \
      --num_cots $NUM_COTS \
      --reasoning_effort "$REASONING_EFFORT"
fi

echo ""
echo "✓ CoT 数据生成完成"
echo ""

# ===== 步骤 3: 评估（Best-of-N）=====
echo "步骤 3: 评估 (Best-of-N 模式)..."
echo "----------------------------------------"

python evaluate_gpt.py \
  --api_key_path "$API_KEY_PATH" \
  --meta_json data/medical_meta.json \
  --eval_mode best_of_n \
  --output_results results_best_of_n.json

echo ""
echo "✓ Best-of-N 评估完成"
echo ""

# ===== 步骤 4: Zero-shot 基线评估（可选）=====
echo "步骤 4: Zero-shot 基线评估..."
echo "----------------------------------------"

if [ $MAX_PATIENTS -gt 0 ]; then
    python evaluate_gpt.py \
      --api_key_path "$API_KEY_PATH" \
      --segmentation_json "$SEGMENTATION_JSON" \
      --eval_mode zero_shot \
      --model gpt-4o \
      --output_results results_zero_shot.json \
      --max_patients $MAX_PATIENTS
else
    python evaluate_gpt.py \
      --api_key_path "$API_KEY_PATH" \
      --segmentation_json "$SEGMENTATION_JSON" \
      --eval_mode zero_shot \
      --model gpt-4o \
      --output_results results_zero_shot.json
fi

echo ""
echo "✓ Zero-shot 评估完成"
echo ""

# ===== 完成 =====
echo "========================================="
echo "所有步骤完成!"
echo "========================================="
echo ""
echo "生成的文件:"
echo "  - segmentation_processed.json: 预处理后的数据"
echo "  - data/medical_train.json: 训练数据（步骤级）"
echo "  - data/medical_meta.json: Meta 数据（完整 CoT）"
echo "  - results_best_of_n.json: Best-of-N 评估结果"
echo "  - results_zero_shot.json: Zero-shot 基线结果"
echo ""
echo "查看评估结果:"
echo "  cat results_best_of_n.json | jq"
echo "  cat results_zero_shot.json | jq"
echo ""
echo "如需训练 PRM，请运行:"
echo "  python main_medical.py \\"
echo "    --train_json_file data/medical_train.json \\"
echo "    --meta_json_file data/medical_meta.json \\"
echo "    --weights_path weights_medical \\"
echo "    --iteration_num 5000"
echo ""
