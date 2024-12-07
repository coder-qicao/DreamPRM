import sys

sys.path.append("/home/q9cao/python_project/multimodal_reasoning")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
path = "/home/q9cao/python_project/multimodal_reasoning"
os.chdir(path)
from utils.json_processor import read_json
import itertools
import numpy as np

dataset = "WeMath"
sequence = read_json(f"BoN/results/{dataset}_internVL_sequence.json")
group_size = 8
selection_size = 4

# regroup
regroup_sequence = []
for j in range(group_size):
    group = []
    for i in range(len(sequence)):
        if i % group_size == j:
            group.append(sequence[i])
    regroup_sequence.append(group)

# selection
combs = list(itertools.combinations(range(group_size), selection_size))
max_accuracy = 0
accuracy_list = []
for k in combs:
    true_num = 0
    false_num = 0
    for l in range(len(regroup_sequence[0])):
        max_score = 0
        for m in k:
            if regroup_sequence[m][l]['mean'] > max_score:
                max_score = regroup_sequence[m][l]['mean']
                true_false = regroup_sequence[m][l]['label']
        if true_false is True:
            true_num += 1
        else:
            false_num += 1
    accuracy = true_num / (true_num + false_num)
    accuracy_list.append(accuracy)
    max_accuracy = max(max_accuracy, accuracy)
    if accuracy == max_accuracy:
        best_comb = k
    print(f"{k}: {accuracy:.4f}, max accuracy: {max_accuracy:.4f}")

print(f"mean : {np.mean(accuracy_list):.4f}, std : {np.std(accuracy_list):.4f}")

# scaling test
for i in range(selection_size):
    combs = list(itertools.combinations(best_comb, i+1))
    min_accuracy = 1
    for k in combs:
        true_num = 0
        false_num = 0
        for l in range(len(regroup_sequence[0])):
            max_score = 0
            for m in k:
                if regroup_sequence[m][l]['mean'] > max_score:
                    max_score = regroup_sequence[m][l]['mean']
                    true_false = regroup_sequence[m][l]['label']
            if true_false is True:
                true_num += 1
            else:
                false_num += 1
        accuracy = true_num / (true_num + false_num)
        min_accuracy = min(min_accuracy, accuracy)
    print(f"CoT num: {i+1}, accuracy: {min_accuracy:.4f}")


