import os
from PIL import Image

def zero_shot_prompt_building_single_image(input, image_path, processor,
                                           tokenize=False, add_generation_prompt=True, type='multi_choice', hint = '',
                                           add = ''):
    placeholder = f"<|image_1|>\n"
    if os.path.isfile(image_path):
        image = [Image.open(image_path)]
        text = input
        e = "Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format \"Final answer: ..\"\n\nQuestion:"
        # hint = "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end. Let's answer step by step.\n\n"
        # e = (
        #     "Question: An administrator at the Department of Motor Vehicles (DMV) tracked the average wait time from month to month. According to the table, what was the rate of change between August and September? (Unit: minutes per month)\nHint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nStep 1: Identify the wait time in August and September.\nThe wait time in August is 17 minutes, and in September it is 14 minutes.\n\nStep 2: Calculate the rate of change.\nThe rate of change is the difference in wait time divided by the number of months.\nRate of change = (14 - 17) / (September - August)\nRate of change = -3 / 1\nRate of change = -3 minutes per month\n\nAnswer: -3\n\n"
        #     "Question: What is the highest amount this class measures?\nHint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\n\nStep 1: Identify the measurement on the glass.\nThe measurement on the glass is 400ml.\n\nStep 2: Determine the highest amount.\nThe highest amount this class measures is 400ml.\n\nAnswer: 400\n\n"
        # )
        query = e + text + add
        messages = [
            {"role": "user", "content": placeholder + query},
        ]

    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt
    )
    return prompt, image

def generate_function_input_single_image(data, dataset, part):
    input = data["input"]
    image_path = f"datasets/{dataset}/images/{part}/{data['id']}.png"
    return input, image_path