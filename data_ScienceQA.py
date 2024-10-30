import json
from datasets import load_dataset

def get_question_text(problem):
    question = problem['question']
    return question


def get_context_text(problem):
    context = problem['hint']
    if context == "":
        context = "N/A"
    return context


def get_image(problem, use_caption, id):
    if use_caption:
        image_url = problem['image']
        address = f"datasets/ScienceQA/images/{id}.png"
        image_url.save(address, format='PNG')


def get_choice_text(probelm, options):
    choices = probelm['choices']
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    #print(choice_txt)
    return choice_txt


def get_answer(problem, options):
    return options[problem['answer']]


def get_lecture_text(problem):
    # \\n: generate the lecture with more tokens.
    lecture = problem['lecture'].replace("\n", "\\n")
    return lecture


def get_solution_text(problem):
    # \\n: generate the solution with more tokens
    solution = problem['solution'].replace("\n", "\\n")
    return solution


def build_prompt(problems, test_qid):
    print(f"saving data {test_qid}")
    options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    if problems['image'] != None:
        use_caption = True
    else:
        use_caption = False
    question = get_question_text(problems)
    context = get_context_text(problems)
    choice = get_choice_text(problems, options)
    answer = get_answer(problems, options)
    lecture = get_lecture_text(problems)
    solution = get_solution_text(problems)
    get_image(problems, use_caption, test_qid)

    input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    ground_truth = answer
    reasoning = f"Because {lecture} {solution}"
    data = {"id":test_qid, "input": input, "ground_truth": ground_truth, "reasoning": reasoning}
    with open("datasets/ScienceQA/text.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    ds = load_dataset("derek-thomas/ScienceQA")
    for i in range(0, 10):
        problem = ds['test'][i]
        build_prompt(problem, i)