import json

# load data
with open("MMPR/train.json", 'r', encoding='utf-8') as f:
    json_list = json.load(f)

print(f"Total samples: {len(json_list)}")
filter_list = []
filter_list_2 = []
filter_list_3 = []

# Difficulty
for data in json_list:
    steps = data['step_id']
    if len(data['add']) > 200*steps and len(data['add']) < 300*steps:
        filter_list.append(data)

print(f"Difficulty selection: {len(filter_list)} samples")

# Quality
for data in filter_list:
    if data['accuracy'] != 0 and data['accuracy'] != 1:
        filter_list_2.append(data)

print(f"Quality selection: {len(filter_list_2)} samples")

# Diversity
Step_diversity = {'1':0, '2':0, '3':0, '4':0, '5':0}
for data in filter_list_2:
    if data['step_id'] == 1:
        Step_diversity['1'] += 1
    if data['step_id'] == 2:
        Step_diversity['2'] += 1
    if data['step_id'] == 3:
        Step_diversity['3'] += 1
    if data['step_id'] == 4:
        Step_diversity['4'] += 1
    if data['step_id'] == 5:
        Step_diversity['5'] += 1
# print(Step_diversity)
for data in filter_list:
    if data['step_id'] in [1, 2, 3, 4, 5]:
        if Step_diversity[str(data['step_id'])] < 2000:
            filter_list_3.append(data)
            Step_diversity[str(data['step_id'])]+=1

print(f"Diversity selection: {len(filter_list_3)} samples")
print(f"Total samples: {len(filter_list_3)+len(filter_list_2)}")





