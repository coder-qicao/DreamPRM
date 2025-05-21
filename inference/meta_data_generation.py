# Written by QI CAO on May 21, 2025.
# All code is original unless otherwise noted.

import sys
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='Project root path')
parser.add_argument('--gpu', type=str, default='0', help='GPU device ID (CUDA_VISIBLE_DEVICES)')
args = parser.parse_args()

# Append project root path to sys.path for module importing
sys.path.append(args.path)

# Set visible CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Change working directory to the project root
os.chdir(args.path)

# Optional: Print confirmation
print(f"CUDA_VISIBLE_DEVICES set to {args.gpu}")
print(f"Working directory changed to {args.path}")

# main

from utils.json_processor import read_json, write_json

dataset = "MMMU"
path = f"inference/results/{dataset}/InternVL-MPO/"
dataset_path = f"dataset/{dataset}/test.json"
d = read_json(dataset_path)
# file = ["0.json","1.json","2.json","3.json","4.json","5.json","6.json","7.json","8.json","9.json","10.json","11.json","12.json","13.json","14.json","15.json",]
file = ["0.json","1.json","2.json","3.json"]
f_list = []
for i in file:
    f = read_json(path+i)
    f_list.append(f)

selected_data = []
selected_data_2 = []
selected_data_3 = []
meta_json = []
for i in range(len(f_list[0])):
    flag = 0
    state = 0
    j_1 = {}
    for j in f_list:
        if state == 0 and j[i]['true_false'] == True:
            state = True
            j_1['id'] = j[i]['id']
            j_1['true_false'] = j[i]['true_false']
            j_1['input'] = j[i]['input']
            j_1['image_path'] = f"dataset/{dataset}/images/test/{j[i]['id']}.png"
        elif state == 0 and j[i]['true_false'] == False:
            state = False
            j_1['id'] = j[i]['id']
            j_1['true_false'] = j[i]['true_false']
            j_1['input'] = j[i]['input']
            j_1['image_path'] = f"dataset/{dataset}/images/test/{j[i]['id']}.png"
        if j[i]['true_false'] != state and "Final answer: " in j[i]["response"]:
            selected_data.append(i)
            state = 0
            image_path = f"dataset/{dataset}/images/test/{j[i]['id']}.png"
            meta_json.append(j_1)
            meta_json.append({'id':j[i]['id'],
                              'true_false':j[i]['true_false'],
                              'input':j[i]['input'],
                              'image_path':image_path})
            break
max_len = 122
true_num = 0
false_num = 0
for i in range(len(f_list[0])):
    flag = 0
    state = 0
    change = 0
    for j in f_list:
        if state == 0 and j[i]['true_false'] == True:
            state = True
        elif state == 0 and j[i]['true_false'] == False:
            state = False
        if j[i]['true_false'] != state:
            change = 1
    if change != 1 and true_num < max_len and state is True:
        true_num += 1
        image_path = f"dataset/{dataset}/images/test/{j[i]['id']}.png"
        meta_json.append({'id': j[i]['id'],
                          'true_false': j[i]['true_false'],
                          'input': j[i]['input'],
                          'image_path': image_path})
        selected_data_2.append(i)
    if change != 1 and false_num < max_len and state is False and "Final answer: " in j[i]["response"]:
        false_num += 1
        image_path = f"dataset/{dataset}/images/test/{j[i]['id']}.png"
        meta_json.append({'id': j[i]['id'],
                          'true_false': j[i]['true_false'],
                          'input': j[i]['input'],
                          'image_path': image_path})
        selected_data_3.append(i)

print(len(selected_data))
print(len(selected_data_2))
print(len(selected_data_3))
print(len(meta_json))
write_json("reweighting/MMPR/InternVL-MPO/meta.json", meta_json)

