import os
import json


def merge_json_files(folder_path, output_file):
    # 初始化一个空列表来存储所有 JSON 文件的内容
    merged_data = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)

            # 打开并读取 JSON 文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 假设每个 JSON 文件是一个字典的列表
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    print(f"Warning: {filename} is not a list of dictionaries and will be skipped.")

    # 将合并后的数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f"All JSON files have been merged into {output_file}")


# 示例用法
folder_path = './'  # 替换为你的文件夹路径
output_file = '0.json'  # 替换为你想要的输出文件名
merge_json_files(folder_path, output_file)