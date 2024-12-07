from utils.json_processor import read_json, write_json
import os
import copy
path = "/home/qi/python_project/multimodal_reasoning"
os.chdir(path)

json_list = ["MCTS/results/MMPR/1000_accuracy.json", "MCTS/results/MMPR/3000_accuracy.json", "MCTS/results/MMPR/5000_accuracy.json"]
data = []
for i in json_list:
    f = read_json(i)
    for sample in f:
        if sample["accuracy"] == 1:
            augmented = copy.deepcopy(sample)
            augmented["accuracy"] = 0
            augmented["image_path"] = "dataset/MMPR/images/white_image_100x100.png"
            f.append(augmented)
    data = data + f

write_json("Reward_model/data/train_augmented.json", data)