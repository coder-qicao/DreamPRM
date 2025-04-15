import json
from datasets import load_dataset
import re

class MathVision_utils:
    def __init__(self):
        self.ds = load_dataset("MathLLMs/MathVision")
        self.data = []

    def save_as_json(self, path="", part="test"):
        ds_part = part
        for i in range(len(self.ds[ds_part])):
            problem = self.ds[ds_part][i]
            self.data.append(self.build_prompt(problem, i))
            if problem['image'] is not None:
                use_caption = True
            else:
                use_caption = False
            self.get_image(problem, use_caption, i, part)
        with open(f"{path}{part}.json", "w") as f:
            json.dump(self.data, f, indent=4)


    def get_question_text(self,problem):
        question = problem['question']
        question = re.sub(r"<image\d+>", "", question)
        return question


    def get_image(self,problem, use_caption, id, part):
        if use_caption:
            image_url = problem['decoded_image']
            address = f"images/{part}/{id}.png"
            image_url.save(address, format='PNG')

    def get_choice_text(self, problem):
        choices = problem['options']
        options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append("({}) {}".format(options[i], c))
        choice_txt = " ".join(choice_list)
        return choice_txt

    def get_answer(self, problem):
        return problem['answer']

    def get_subject(self, problem):
        return problem['subject']

    def build_prompt(self, problems, test_qid):
        print(f"saving data {test_qid}")
        question = self.get_question_text(problems)
        if problems["options"] != []:
            choice = self.get_choice_text(problems)
            input = f"Question: {question}\nOptions: {choice}\n"
        else:
            input = f"Question: {question}\n"
        answer = self.get_answer(problems)
        subject = self.get_subject(problems)
        ground_truth = answer
        return {"id":test_qid, "input": input, "subject": subject, "ground_truth": ground_truth}


if __name__ == "__main__":
    ds = MathVision_utils()
    ds.save_as_json(path="", part="test")