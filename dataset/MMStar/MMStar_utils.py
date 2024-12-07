import json
from datasets import load_dataset

class MMStar_utils:
    def __init__(self):
        self.ds = load_dataset("Lin-Chen/MMStar")
        self.data = []

    def save_as_json(self, path="", part="test"):
        if part == "test":
            ds_part = "val"
        else:
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
        return question


    def get_image(self,problem, use_caption, id, part):
        if use_caption:
            image_url = problem['image']
            address = f"images/{part}/{id}.png"
            image_url.save(address, format='PNG')

    def get_answer(self, problem):
        return problem['answer']

    def get_category(self, problem):
        return problem['category']

    def build_prompt(self, problems, test_qid):
        print(f"saving data {test_qid}")
        question = self.get_question_text(problems)
        answer = self.get_answer(problems)
        category = self.get_category(problems)
        input = f"Question: {question}\n"
        ground_truth = answer
        return {"id":test_qid, "input": input, "category": category, "ground_truth": ground_truth}


if __name__ == "__main__":
    ds = MMStar_utils()
    ds.save_as_json(path="", part="test")