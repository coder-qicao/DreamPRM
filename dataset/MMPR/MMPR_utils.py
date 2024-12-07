import json
from utils.json_processor import read_json
from datasets import load_dataset
import random
import os

path = "/home/qi/python_project/multimodal_reasoning"
os.chdir(path)

class MMPR_utils:
    def __init__(self, dataset_list, max_num):
        self.dataset = []
        for d in dataset_list:
            dn = "dataset/MMPR/annotations/" + d
            with open(dn, "r", encoding="utf-8") as f:
                ds = [json.loads(line) for line in f]
            ds = random.sample(ds, max_num)
            for x in ds:
                x['dataset'] = d.replace(".jsonl",'')
            self.dataset += ds
        self.meta = read_json("dataset/MMPR/meta.json")
        self.data = []

    def save_as_json(self, path=""):
        index = 0
        for problem in self.dataset:
            self.data.append(self.build_prompt(problem, index))
            index += 1
            print(f"data {index}")
        with open(f"dataset/MMPR/train.json", "w") as f:
            json.dump(self.data, f, indent=4)

    def get_question_text(self,problem):
        question = problem['question']
        return question

    def get_image_path(self,problem):
        dataset = problem['dataset']
        image_url = problem['image']
        root_dir = self.meta[dataset]['root']
        root_dir = "dataset/" + root_dir
        return root_dir + '/' + image_url

    def get_answer(self,problem):
        answer = problem['answer_gt']
        return answer

    def get_dataset(self,problem):
        dataset = problem['dataset']
        return dataset

    def build_prompt(self, problems, test_qid):
        question = self.get_question_text(problems)
        image_path = self.get_image_path(problems)
        answer = self.get_answer(problems)
        dataset = self.get_dataset(problems)
        return {"id":test_qid, "input": question, "image_path": image_path, "ground_truth": answer, "dataset": dataset}


if __name__ == "__main__":
    ds_json = ["ai2d_train_12k_en_20240410_extracted_pairs_vqa_correctness_rules.jsonl",
               "chartqa_trainval_30k_w_csv_en_20240402_extracted_pairs_vqa_correctness_rules.jsonl",
               "m3cot_train_extracted_pairs_vqa_correctness_rules.jsonl",
               "scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_correctness_rules.jsonl",
               "mapqa_suv_en_20240402_extracted_pairs_vqa_correctness_rules.jsonl",
               "geo170k_extracted_full_pairs_vqa_correctness_rules.jsonl",
               "CLEVR_math_en_20240402_extracted_pairs_vqa_correctness_rules.jsonl",
               "geometry3k_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules.jsonl",
               "figureqa_en_20240402_extracted_pairs_vqa_correctness_rules.jsonl",
               "infographics_20240403_qa_20240407_v2_extracted_pairs_vqa_correctness_rules.jsonl"]
    ds = MMPR_utils(ds_json, 1000)
    ds.save_as_json(path="")