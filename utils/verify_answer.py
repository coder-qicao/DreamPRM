import re

def verify_answer_multi_choice(answer, ground_truth):
    matches = re.findall(r'(?<!Answer)([ABCD])', answer)
    answer = matches[-1] if matches else None  # 获取最后一个匹配项

    if answer == ground_truth:
        return True
    else:
        return False


def verify_answer(answer, ground_truth, question_type):
    if question_type == 'multi_choice':
        return verify_answer_multi_choice(answer, ground_truth)
    else:
        if "Final answer: " in answer:
            answer = answer.split("Final answer: ")[-1]
            if answer == ground_truth:
                return True
            else:
                return False
        else:
            return False
