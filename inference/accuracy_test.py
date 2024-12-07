from utils.json_processor import read_json

file_path = "results/Phi_vision.json"
data = read_json(file_path)

# original accuracy
true_num = 0
false_num = 0
for i in data:
    flag = i['id']
    if i['true_false'] is True:
        true_num += 1
    else:
        false_num += 1
print(f"Original Accuracy: {true_num / (true_num + false_num)}")
