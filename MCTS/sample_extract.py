import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
path = "/home/q9cao/python_project/multimodal_reasoning"
os.chdir(path)
import json
import os
import shutil
from urllib.request import urlretrieve


def extract_and_save_images(json_data, target_id, target_sid, output_folder):
    """
    从JSON数据中提取指定id和sid的数据集名称和图片，并保存图片到指定文件夹

    Args:
        json_data: 包含所有数据的JSON对象
        target_id: 要提取的id
        target_sid: 要提取的sid
        output_folder: 图片保存的目标文件夹
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 查找匹配的数据项
    matched_items = []
    for item in json_data:
        if item.get("id") == target_id and item.get("sid") == target_sid:
            matched_items.append(item)

    if not matched_items:
        print(f"未找到id={target_id}, sid={target_sid}的数据")
        return

    # 处理每个匹配项
    for item in matched_items:
        dataset_name = item.get("dataset", "未知数据集")
        image_path = item.get("image_path")
        input = item.get("input")
        add = item.get("add")
        score = item.get("accuracy")

        if not image_path:
            print(f"数据项{target_id}-{target_sid}没有图片路径")
            continue

        # 提取文件名
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, filename)

        # 复制或下载图片
        try:
            if image_path.startswith(("http://", "https://")):
                # 如果是网络图片，下载它
                print(f"正在下载图片: {image_path}")
                urlretrieve(image_path, output_path)
            else:
                # 如果是本地路径，复制文件
                print(f"正在复制图片: {image_path}")
                shutil.copy2(image_path, output_path)

            print(f"成功保存图片到: {output_path}")
            print(f"数据集名称: {dataset_name}")
            print(f"输入：{input}")
            print(f"输出：{add}")
            print(f"分数：{score}")
        except Exception as e:
            print(f"处理图片时出错: {e}")


# 示例用法
if __name__ == "__main__":
    # 加载JSON数据
    with open("MCTS/results/MMPR/Tree/0.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 指定要提取的id和sid
    target_id = 5400
    target_sid = 3

    # 指定输出文件夹
    output_folder = "MCTS/results/MMPR/samples"

    # 调用函数
    extract_and_save_images(data, target_id, target_sid, output_folder)