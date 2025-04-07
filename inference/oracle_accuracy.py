from utils.json_processor import read_json
import os

path = "/home/q9cao/python_project/multimodal_reasoning"
os.chdir(path)

dataset = "WeMath"
path = f"inference/results/{dataset}/InternVL-MPO/"
# file = ["0.json","1.json","2.json","3.json","4.json","5.json","6.json","7.json","8.json","9.json","10.json","11.json","12.json","13.json","14.json","15.json",]
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