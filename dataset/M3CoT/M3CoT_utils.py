import json
from datasets import load_dataset

class M3CoT_utils:
    def __init__(self):
        self.ds = load_dataset("LightChen2333/M3CoT")
        self.data = []

    def save_as_json(self, path="", part="test"):
        for i in range(len(self.ds[part])):
            problem = self.ds[part][i]
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

    def get_choice_text(self,probelm, options):
        if options:
            choices = probelm['choices']
            choice_list = []
            for i, c in enumerate(choices):
                choice_list.append("({}) {}".format(options[i], c))
            choice_txt = " ".join(choice_list)
            #print(choice_txt)
            return choice_txt
        else:
            return None

    def get_answer(self, problem, options):
        return problem['answer']

    def get_domain(self, problem):
        return problem['domain']

    def build_prompt(self, problems, test_qid):
        print(f"saving data {test_qid}")
        question = self.get_question_text(problems)
        options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        choice = self.get_choice_text(problems, options)
        answer = self.get_answer(problems, options)
        domain = self.get_domain(problems)
        input = f"Question: {question}\nOptions: {choice}\n"
        ground_truth = answer
        return {"id":test_qid, "input": input, "domain": domain, "ground_truth": ground_truth}


if __name__ == "__main__":
    ds = M3CoT_utils()
    ds.save_as_json(path="", part="test")