"""
测试脚本 - 验证医学影像分析环境设置
检查所有必要的文件和依赖是否正确配置
"""

import os
import sys
import json
from pathlib import Path

def print_status(message, status):
    """打印带颜色的状态信息"""
    if status == 'ok':
        print(f"✓ {message}")
    elif status == 'warning':
        print(f"⚠ {message}")
    elif status == 'error':
        print(f"✗ {message}")
    else:
        print(f"  {message}")

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print_status(f"{description}: {filepath}", 'ok')
        return True
    else:
        print_status(f"{description} 不存在: {filepath}", 'error')
        return False

def check_python_modules():
    """检查 Python 模块依赖"""
    print("\n=== 检查 Python 模块 ===")
    modules = [
        'torch',
        'numpy',
        'requests',
        'PIL',
        'transformers',
        'betty'
    ]

    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print_status(f"{module}", 'ok')
        except ImportError:
            print_status(f"{module} 未安装", 'error')
            all_ok = False

    return all_ok

def check_files():
    """检查必要文件"""
    print("\n=== 检查必要文件 ===")

    files_to_check = [
        ('generate_cot_segmentation.py', 'CoT 生成脚本'),
        ('preprocess_segmentation.py', '数据预处理脚本'),
        ('evaluate_gpt.py', '评估脚本'),
        ('main_medical.py', '医学任务训练脚本'),
        ('README_MEDICAL.md', '医学任务说明文档'),
        ('config_medical.yaml', '配置文件'),
    ]

    all_ok = True
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_ok = False

    return all_ok

def check_data_files():
    """检查数据文件"""
    print("\n=== 检查数据文件 ===")

    # 检查 API key
    if check_file_exists('api.txt', 'API key 文件'):
        with open('api.txt', 'r') as f:
            key = f.read().strip()
            if key and len(key) > 10:
                print_status(f"  API key 长度: {len(key)} 字符", 'ok')
            else:
                print_status(f"  API key 似乎无效（长度: {len(key)}）", 'warning')
    else:
        print_status("  请创建 api.txt 并写入你的 API key", 'info')

    # 检查 segmentation.json
    if check_file_exists('segmentation.json', 'Segmentation 数据'):
        try:
            with open('segmentation.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                print_status(f"  包含 {len(data)} 个患者数据", 'ok')

                # 检查第一个患者的数据结构
                if data:
                    patient = data[0]
                    required_fields = [
                        'patient',
                        'Image description',
                        'image diagnosis',
                        'Lymphatic_node_transfer',
                        'Invade the uterus',
                        'Invade the vagina',
                        'segmentation_images'
                    ]

                    missing_fields = [f for f in required_fields if f not in patient]
                    if missing_fields:
                        print_status(f"  警告: 第一个患者缺少字段: {missing_fields}", 'warning')
                    else:
                        print_status(f"  数据结构完整", 'ok')

                    # 检查图片路径
                    if 'segmentation_images' in patient:
                        images = patient['segmentation_images']
                        print_status(f"  第一个患者有 {len(images)} 张图片", 'ok')

                        # 检查第一张图片是否存在
                        if images:
                            first_image = images[0]
                            if os.path.exists(first_image):
                                print_status(f"  第一张图片存在: {first_image}", 'ok')
                            else:
                                print_status(f"  第一张图片不存在: {first_image}", 'error')

        except json.JSONDecodeError as e:
            print_status(f"  JSON 解析错误: {e}", 'error')
            return False
        except Exception as e:
            print_status(f"  读取错误: {e}", 'error')
            return False
    else:
        print_status("  segmentation.json 是运行流程的必要文件", 'info')
        return False

    # 检查数据目录
    if not os.path.exists('data'):
        print_status("创建 data/ 目录...", 'info')
        os.makedirs('data', exist_ok=True)

    return True

def check_directory_structure():
    """检查目录结构"""
    print("\n=== 检查目录结构 ===")

    directories = ['data', 'weights_medical', 'evaluation_results']

    for directory in directories:
        if os.path.exists(directory):
            print_status(f"{directory}/ 目录存在", 'ok')
        else:
            print_status(f"创建 {directory}/ 目录", 'info')
            os.makedirs(directory, exist_ok=True)

def run_simple_test():
    """运行简单的功能测试"""
    print("\n=== 运行简单测试 ===")

    # 测试 image_to_base64 函数
    try:
        import base64
        test_string = "test"
        encoded = base64.b64encode(test_string.encode()).decode()
        decoded = base64.b64decode(encoded).decode()
        if decoded == test_string:
            print_status("Base64 编码/解码正常", 'ok')
        else:
            print_status("Base64 编码/解码失败", 'error')
    except Exception as e:
        print_status(f"Base64 测试失败: {e}", 'error')

    # 测试 JSON 读写
    try:
        test_data = {'test': 'data'}
        test_file = 'test_temp.json'
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        with open(test_file, 'r') as f:
            loaded = json.load(f)
        os.remove(test_file)
        if loaded == test_data:
            print_status("JSON 读写正常", 'ok')
        else:
            print_status("JSON 读写失败", 'error')
    except Exception as e:
        print_status(f"JSON 测试失败: {e}", 'error')

def print_next_steps():
    """打印下一步操作指南"""
    print("\n" + "="*60)
    print("下一步操作指南")
    print("="*60)

    print("\n1. 准备数据:")
    print("   - 确保 segmentation.json 在项目根目录")
    print("   - 确保 api.txt 包含有效的 API key")

    print("\n2. 运行完整流程:")
    print("   bash run_medical_pipeline.sh")

    print("\n3. 或者分步运行:")
    print("   # 步骤 1: 数据预处理")
    print("   python preprocess_segmentation.py --segmentation_json segmentation.json --select_valid_images")

    print("\n   # 步骤 2: 生成 CoT")
    print("   python generate_cot_segmentation.py --segmentation_json segmentation.json --max_patients 10")

    print("\n   # 步骤 3: 评估")
    print("   python evaluate_gpt.py --meta_json data/medical_meta.json --eval_mode best_of_n")

    print("\n4. 查看文档:")
    print("   cat README_MEDICAL.md")

    print("\n" + "="*60)

def main():
    """主函数"""
    print("="*60)
    print("医学影像分析环境检查")
    print("="*60)

    all_ok = True

    # 检查 Python 版本
    print(f"\nPython 版本: {sys.version}")
    if sys.version_info < (3, 8):
        print_status("警告: Python 版本过低，建议使用 3.8+", 'warning')
        all_ok = False

    # 检查各个组件
    if not check_python_modules():
        all_ok = False

    if not check_files():
        all_ok = False

    if not check_data_files():
        all_ok = False

    check_directory_structure()
    run_simple_test()

    # 打印总结
    print("\n" + "="*60)
    if all_ok:
        print("✓ 环境检查完成，一切正常!")
        print("="*60)
        print_next_steps()
    else:
        print("⚠ 环境检查发现一些问题，请根据上面的提示进行修复")
        print("="*60)

if __name__ == '__main__':
    main()
