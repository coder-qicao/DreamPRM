import json
from datasets import load_dataset

class WeMath_utils:
    def __init__(self):
        self.ds = load_dataset("We-Math/We-Math")
        self.data = []

    def save_as_json(self, path="", part="test"):
        if part == "test":
            ds_part = "testmini"
        else:
            ds_part = part
        for i in range(len(self.ds[ds_part])):
            problem = self.ds[ds_part][i]
            self.data.append(self.build_prompt(problem, i))
            if problem['image_path'] is not None:
                use_caption = True
            else:
                use_caption = False
            self.get_image(problem, use_caption, i, part)
        with open(f"{path}{part}.json", "w") as f:
            json.dump(self.data, f, indent=4)


    def get_question_text(self,problem):
        question = problem['question']
        return question


    def get_image(self,problem, use_caption, id, part):
        if use_caption:
            image_url = problem['image_path']
            address = f"images/{part}/{id}.png"
            image_url.save(address, format='PNG')

    def get_choice_text(self,problem):
        return problem['option']

    def get_answer(self, problem):
        return problem['answer']

    def get_key(self, problem):
        return problem['key']

    def build_prompt(self, problems, test_qid):
        print(f"saving data {test_qid}")
        question = self.get_question_text(problems)
        choice = self.get_choice_text(problems)
        answer = self.get_answer(problems)
        key = self.get_key(problems)
        input = f"Question: {question}\nOptions: {choice}\n"
        ground_truth = answer
        return {"id":test_qid, "input": input, "key": key, "ground_truth": ground_truth}


if __name__ == "__main__":
    ds = WeMath_utils()
    ds.save_as_json(path="", part="test")