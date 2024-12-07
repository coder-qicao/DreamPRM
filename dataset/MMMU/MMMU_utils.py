import json
from datasets import load_dataset
import ast

class MMMU_utils:
    def __init__(self):
        self.ds = load_dataset("MMMU/MMMU_Pro", "standard (4 options)")
        self.data = []

    def save_as_json(self, path="", part="test"):
        if part == "test":
            ds_part = "test"
        else:
            ds_part = part
        id = 0
        for i in range(len(self.ds[ds_part])):
            problem = self.ds[ds_part][i]
            if problem['image_2'] is not None or problem['question'].startswith("<image 1>"):
                continue
            if not problem['question'].endswith("<image 1>"):
                continue
            self.data.append(self.build_prompt(problem, id))
            id += 1
            if problem['image_1'] is not None:
                use_caption = True
            else:
                use_caption = False
            self.get_image(problem, use_caption, i, part)
        with open(f"{path}{part}.json", "w") as f:
            json.dump(self.data, f, indent=4)


    def get_question_text(self,problem):
        question = problem['question']
        if question.endswith("<image 1>"):
            question = question[:-len("<image 1>")]
        return question


    def get_image(self,problem, use_caption, id, part):
        if use_caption:
            image_url = problem['image_1']
            address = f"images/{part}/{id}.png"
            image_url.save(address, format='PNG')

    def get_answer(self, problem):
        return problem['answer']

    def get_choice_text(self, problem, options):
        choices = problem['options']
        choice_list = []
        choices = ast.literal_eval(choices)
        for i, c in enumerate(choices):
            choice_list.append("({}) {}".format(options[i], c))
        choice_txt = " ".join(choice_list)
        # print(choice_txt)
        return choice_txt

    def build_prompt(self, problems, test_qid):
        print(f"saving data {test_qid}")
        question = self.get_question_text(problems)
        answer = self.get_answer(problems)
        choice_txt = self.get_choice_text(problems, ['A','B','C','D','E','F','G','H','I','J','K','L','M','N'])
        input = f"Question: {question}\nOptions: {choice_txt}"
        ground_truth = answer
        return {"id":test_qid, "input": input, "ground_truth": ground_truth}


if __name__ == "__main__":
    ds = MMMU_utils()
    ds.save_as_json(path="", part="test")