import json
from datasets import load_dataset

class ScienceQA:
    def __init__(self):
        self.ds = load_dataset("derek-thomas/ScienceQA")
        self.data = []

    def save_as_json(self, path="datasets/ScienceQA/", part="test"):
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

    def get_context_text(self,problem):
        context = problem['hint']
        if context == "":
            context = "N/A"
        return context

    def get_image(self,problem, use_caption, id, part):
        if use_caption:
            image_url = problem['image']
            address = f"datasets/ScienceQA/images/{part}/{id}.png"
            image_url.save(address, format='PNG')

    def get_choice_text(self,probelm, options):
        choices = probelm['choices']
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append("({}) {}".format(options[i], c))
        choice_txt = " ".join(choice_list)
        #print(choice_txt)
        return choice_txt

    def get_answer(self,problem, options):
        return options[problem['answer']]

    def get_lecture_text(self,problem):
        # \\n: generate the lecture with more tokens.
        lecture = problem['lecture'].replace("\n", "\\n")
        return lecture

    def get_solution_text(self,problem):
        # \\n: generate the solution with more tokens
        solution = problem['solution'].replace("\n", "\\n")
        return solution

    def build_prompt(self, problems, test_qid):
        print(f"saving data {test_qid}")
        options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

        question = self.get_question_text(problems)
        context = self.get_context_text(problems)
        choice = self.get_choice_text(problems, options)
        answer = self.get_answer(problems, options)
        lecture = self.get_lecture_text(problems)
        solution = self.get_solution_text(problems)

        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
        ground_truth = answer
        reasoning = f"Because {lecture} {solution}"
        return {"id":test_qid, "input": input, "ground_truth": ground_truth, "reasoning": reasoning}


if __name__ == "__main__":
    ds = ScienceQA()
    ds.save_as_json(path="datasets/ScienceQA/", part="train")