"""
数据预处理脚本 - 验证和准备 segmentation.json 数据
用于检查数据完整性，选择有效图片，生成数据统计
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

# ===== 配置参数 =====
parser = argparse.ArgumentParser(description="Preprocess segmentation data")
parser.add_argument('--segmentation_json', type=str, required=True, help='Path to segmentation JSON file')
parser.add_argument('--output_json', type=str, default='segmentation_processed.json', help='Output processed JSON')
parser.add_argument('--check_images', action='store_true', help='Check if all images exist and are valid')
parser.add_argument('--select_valid_images', action='store_true', help='Select valid PNG images for each patient')

args = parser.parse_args()

def check_image_validity(image_path: str) -> Dict[str, Any]:
    """
    检查图片是否有效

    Returns:
        Dict with 'exists', 'valid', 'size', 'format'
    """
    result = {
        'exists': False,
        'valid': False,
        'size': None,
        'format': None,
        'error': None
    }

    if not os.path.exists(image_path):
        result['error'] = 'File not found'
        return result

    result['exists'] = True

    try:
        img = Image.open(image_path)
        result['size'] = img.size
        result['format'] = img.format
        result['valid'] = True
        img.close()
    except Exception as e:
        result['error'] = str(e)

    return result

def select_valid_png(segmentation_images: List[str]) -> Dict[str, Any]:
    """
    从 segmentation_images 中选择有效的 PNG 图片

    Returns:
        Dict with 'selected_image', 'total_images', 'valid_images'
    """
    valid_images = []

    for img_path in segmentation_images:
        if not img_path.endswith('.png'):
            continue

        check_result = check_image_validity(img_path)
        if check_result['valid']:
            valid_images.append({
                'path': img_path,
                'size': check_result['size'],
                'format': check_result['format']
            })

    return {
        'selected_image': valid_images[0]['path'] if valid_images else None,
        'total_images': len(segmentation_images),
        'valid_png_images': len(valid_images),
        'valid_images_info': valid_images
    }

def normalize_label(label: str) -> str:
    """标准化标签值为 yes/no"""
    label_lower = str(label).lower().strip()
    if label_lower in ['yes', 'y', '1', 'true']:
        return 'yes'
    elif label_lower in ['no', 'n', '0', 'false']:
        return 'no'
    else:
        return 'unknown'

def preprocess_segmentation_data():
    """预处理 segmentation.json 数据"""

    print(f"读取数据文件: {args.segmentation_json}")
    with open(args.segmentation_json, 'r', encoding='utf-8') as f:
        patients_data = json.load(f)

    print(f"找到 {len(patients_data)} 个患者数据\n")

    # 统计信息
    stats = {
        'total_patients': len(patients_data),
        'patients_with_images': 0,
        'patients_with_valid_images': 0,
        'total_images': 0,
        'valid_images': 0,
        'label_distribution': {
            'Lymphatic_node_transfer': {'yes': 0, 'no': 0, 'unknown': 0},
            'Invade the uterus': {'yes': 0, 'no': 0, 'unknown': 0},
            'Invade the vagina': {'yes': 0, 'no': 0, 'unknown': 0}
        },
        'missing_fields': {
            'Image description': 0,
            'image diagnosis': 0,
            'segmentation_images': 0
        }
    }

    processed_patients = []

    # 处理每个患者
    for patient_idx, patient in enumerate(patients_data):
        patient_id = patient.get('patient', f'patient_{patient_idx}')
        print(f"处理患者 {patient_idx + 1}/{len(patients_data)}: {patient_id}")

        # 创建处理后的患者数据
        processed_patient = {
            'patient': patient_id,
            'age': patient.get('age', None),
            'Image description': patient.get('Image description', ''),
            'image diagnosis': patient.get('image diagnosis', ''),
        }

        # 检查必要字段
        for field in ['Image description', 'image diagnosis', 'segmentation_images']:
            if not patient.get(field):
                stats['missing_fields'][field] += 1
                print(f"  警告: 缺少字段 '{field}'")

        # 处理标签
        labels = {}
        for label_name in ['Lymphatic_node_transfer', 'Invade the uterus', 'Invade the vagina']:
            raw_label = patient.get(label_name, 'unknown')
            normalized_label = normalize_label(raw_label)
            labels[label_name] = normalized_label
            stats['label_distribution'][label_name][normalized_label] += 1

        processed_patient['labels'] = labels

        # 处理图片
        segmentation_images = patient.get('segmentation_images', [])
        if segmentation_images:
            stats['patients_with_images'] += 1
            stats['total_images'] += len(segmentation_images)

        if args.select_valid_images and segmentation_images:
            # 选择有效的 PNG 图片
            selection_result = select_valid_png(segmentation_images)
            processed_patient['selected_image'] = selection_result['selected_image']
            processed_patient['image_selection_info'] = selection_result

            if selection_result['selected_image']:
                stats['patients_with_valid_images'] += 1
                stats['valid_images'] += selection_result['valid_png_images']
                print(f"  选择图片: {selection_result['selected_image']}")
            else:
                print(f"  警告: 没有找到有效的 PNG 图片")
        else:
            # 不检查，直接保留原始列表
            processed_patient['segmentation_images'] = segmentation_images

        # 如果需要检查图片有效性
        if args.check_images and segmentation_images:
            image_checks = []
            for img_path in segmentation_images:
                check_result = check_image_validity(img_path)
                image_checks.append({
                    'path': img_path,
                    'check': check_result
                })

                if not check_result['exists']:
                    print(f"  错误: 图片不存在 - {img_path}")
                elif not check_result['valid']:
                    print(f"  错误: 图片无效 - {img_path}: {check_result['error']}")

            processed_patient['image_checks'] = image_checks

        processed_patients.append(processed_patient)

    # 保存处理后的数据
    print(f"\n保存处理后的数据到: {args.output_json}")
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(processed_patients, f, indent=2, ensure_ascii=False)

    # 打印统计信息
    print("\n" + "="*60)
    print("数据统计")
    print("="*60)
    print(f"总患者数: {stats['total_patients']}")
    print(f"有图片的患者数: {stats['patients_with_images']}")
    if args.select_valid_images:
        print(f"有有效 PNG 图片的患者数: {stats['patients_with_valid_images']}")
    print(f"总图片数: {stats['total_images']}")
    if args.select_valid_images:
        print(f"有效 PNG 图片数: {stats['valid_images']}")

    print("\n标签分布:")
    for label_name, distribution in stats['label_distribution'].items():
        total = sum(distribution.values())
        print(f"  {label_name}:")
        for value, count in distribution.items():
            percentage = count / total * 100 if total > 0 else 0
            print(f"    {value}: {count} ({percentage:.1f}%)")

    print("\n缺失字段统计:")
    for field, count in stats['missing_fields'].items():
        percentage = count / stats['total_patients'] * 100
        print(f"  {field}: {count} ({percentage:.1f}%)")

    print("\n" + "="*60)

if __name__ == '__main__':
    print("=== 医学影像数据预处理脚本 ===\n")
    preprocess_segmentation_data()
    print("\n脚本执行完成!")
