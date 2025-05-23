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
from utils.json_processor import read_json

dataset = "WeMath"
path = f"inference/results/{dataset}/InternVL-MPO/"
# file = ["0.json","1.json","2.json","3.json","4.json","5.json","6.json","7.json","8.json"]
file = ["0.json","1.json","2.json","3.json"]
f_list = []
for i in file:
    f = read_json(path+i)
    f_list.append(f)

true_num = 0
for i in range(len(f_list[0])):
    flag = False
    for j in f_list:
        if j[i]['true_false'] == True:
            flag = True
            break
    if flag:
        true_num += 1

print(f"Accuracy: {true_num/len(f_list[0])}")
print(len(f_list[0]))