import json

def read_json(source):
    with open(source, 'r', encoding='utf-8') as f:
        json_list = json.load(f)
    return json_list

def write_json(source, json_list):
    with open(source, 'w', encoding='utf-8') as f:
        json.dump(json_list, f)