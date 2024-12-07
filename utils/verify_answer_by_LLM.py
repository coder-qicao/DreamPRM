#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
利用开源小型 LLM 模型判定 VQA 任务的答案是否正确
注意：请先安装 transformers 库，例如：pip install transformers torch
"""

def judge_vqa_answer(question, ground_truth, candidate_answer, model, tokenizer):
    """
    利用指定的开源小LLM模型，根据题目、标准答案和待判定答案生成判断结果。

    参数:
      question (str): VQA 题目描述
      ground_truth (str): 标准答案（ground truth）
      candidate_answer (str): 待判断的答案
      model_name (str): 模型名称（例如：'gpt2'，或其它支持指令任务的模型）
      max_length (int): 生成文本的最大长度

    返回:
      str: 模型生成的判断结果文本
    """
    # 构造 prompt，要求模型判断答案是否正确，并给出简要解释
    prompt = (
        "Task: Do not attempt to solve the question. Your only task is to verify if the candidate answer "
        "matches the ground truth. For multiple-choice questions (when options are provided), first check that "
        "the candidate answer is one of the given options and then verify it exactly matches the ground truth. "
        "For non-multiple-choice questions, simply compare the candidate answer with the ground truth"
        "The candidate answer may not be exactly the same as the ground truth. "
        "If they indicate the same thing, the candidate answer should be right."
        # "Follow the <AND> and <OR> logical rules in the ground truth answer."
        f"Input: {question}\n"
        f"Ground truth answer: {ground_truth}\n"
        f"Candidate answer: {candidate_answer}\n\n"
        "If the candidate answer is correct, respond with 'correct'"
        "If it is incorrect, respond with 'incorrect'. "
        "Your response must be a single line with only 'correct' or 'incorrect, and no extra text." )

    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def main():
    # 设置模型名称
    # 注意：这里示例使用的是 "gpt2"，实际推荐使用指令调优过的小模型
    model_name = "gpt2"  # 例如可替换为 "facebook/opt-350m" 或其它模型

    # 示例 VQA 题目及答案
    question = "图片中显示的是一只猫还是一只狗？"
    ground_truth = "猫"
    candidate_answer = "猫"

    output = judge_vqa_answer(question, ground_truth, candidate_answer, model_name)
    print("模型判断结果:")
    print(output)


if __name__ == "__main__":
    main()
