from utils.json_processor import read_json

file_path = "results/MathVista.json"
data = read_json(file_path)
data = data[0:102]

# original accuracy
flag = -1
true_num = 0
false_num = 0
for i in data:
    if i['id'] != flag:
        flag = i['id']
        if i['true_false'] is True:
            true_num += 1
        else:
            false_num += 1
print(f"Original Accuracy: {true_num / (true_num + false_num)}")


# majority vote accuracy
flag = -1
true_num = 0
false_num = 0
group_true_num = 0
group_false_num = 0
for i in data:
    if i['id'] != flag:
        flag = i['id']
        if group_true_num > group_false_num:
            true_num += 1
        elif group_false_num > group_true_num:
            false_num += 1
        group_true_num = 0
        group_false_num = 0
    if i['true_false'] is True:
        group_true_num += 1
    else:
        group_false_num += 1
true_num += 1

print(f"Majority Accuracy: {true_num / (true_num + false_num)}")

# last vote accuracy
flag = -1
true_num = 0
false_num = 0
prev = 0
for i in data:
    if i['id'] != flag:
        flag = i['id']
        if prev is True:
            true_num += 1
        elif prev is False:
            false_num += 1
    if i['true_false'] is True:
        prev = True
    else:
        prev = False
true_num += 1
print(f"Last Accuracy: {true_num / (true_num + false_num)}")

# upperbound accuracy
flag = -1
true_false = 0
true_num = 0
false_num = 0
for i in data:
    if i['id'] != flag:
        flag = i['id']
        if true_false is True:
            true_num += 1
        elif true_false is False:
            false_num += 1
        true_false = False
    if i['true_false'] is True:
        true_false = True
true_num += 1
print(f"Upperbound Accuracy: {true_num / (true_num + false_num)}")
