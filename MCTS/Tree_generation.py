import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
path = "/home/q9cao/python_project/multimodal_reasoning"
os.chdir(path)

from utils.json_processor import read_json, write_json
from utils.split_step import split_step

InternVL_MPO_json = "inference/results/MMPR/InternVL-MPO/10.json"
InternVL_MPO_data = read_json(InternVL_MPO_json)
dataset_json = "dataset/MMPR/train.json"
dataset = read_json(dataset_json)

# 去掉没有final answer的
format_rule_candidate = []
for i in InternVL_MPO_data:
    if 'Final answer: ' in i['response']:
        format_rule_candidate.append(i)
print(len(format_rule_candidate))

# 建立树结构
tree = []
for i in format_rule_candidate:
    for j in range(5):
        id = i['id']
        sid = j + 1
        input = i['input']
        add = split_step(sid, i['response'])
        ground_truth = dataset[id]['ground_truth']
        image_path = dataset[id]['image_path']
        dataset_name = dataset[id]['dataset']
        if i['true_false']:
            score = 1
        else:
            score = 0
        tree.append({'id': id,
                     'sid': sid,
                     'input': input,
                     'add': add,
                     'ground_truth': ground_truth,
                     'image_path': image_path,
                     'dataset': dataset_name,
                     'score': score,
                     'times': 1,
                     'accuracy': score})
print(len(tree))
print(tree[1000])

write_json("MCTS/results/MMPR/Tree/1.json", tree)

