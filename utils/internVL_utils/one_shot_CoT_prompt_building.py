import os
from PIL import Image

def one_shot_prompt_building_single_image(input, hint = '', add = ''):
    placeholder = f"<image>\n"
    text = input
    if hint != '':
        hint = "\n" + hint
    e = """
You have been given a question that involves both an image and a text. 
Your task is to analyze the question by following exactly five steps:

Step 1: Restate the question.
   - Clearly rephrase or clarify the question in your own words.

Step 2: Gather evidence from the image.
   - Describe any relevant visual details (e.g., objects, people, locations, interactions) that might address the question.

Step 3: Identify any background knowledge needed.
   - Note any general facts, assumptions, or external knowledge that is necessary to address the question.

Step 4: Reason with the current evidence.
   - Integrate the information from the image, text, and relevant background knowledge. 
   - Show how these pieces of evidence lead toward an answer.

Step 5: Summarize and conclude with all the information.
   - Provide a concise, direct answer to the question, referencing the supporting evidence and reasoning.
   
Once you have completed your reasoning, provide your final answer in the format:

Final answer: ...

Example:

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format \"Final answer: ..\"

Question:

We have an image showing a person wearing a white lab coat standing in front of a microscope on a lab bench.
The text states: "Dr. Park is a research scientist who works with microscopes and chemicals to study microorganisms."
Which of the following statements is most likely true about Dr. Park?
(A) Dr. Park is a chef who specializes in baking.
(B) Dr. Park is a scientist who probably conducts experiments.
(C) Dr. Park is a construction worker.
(D) Dr. Park is an airline pilot.

Step 1: Restate the question.
   - The question is asking which of the four statements is most likely true about Dr. Park, based on the provided image and text.

Step 2: Gather evidence from the image.
   - The image shows a person wearing a white lab coat, standing in front of a microscope on a lab bench.
   - This setting suggests a laboratory environment and a scientific context.

Step 3: Identify any background knowledge needed.
   - Generally, someone wearing a lab coat and using a microscope is likely engaged in scientific research or lab work.
   - Dr. Park is mentioned in the text as a researcher who studies microorganisms.

Step 4: Reason with the current evidence.
   - The text states Dr. Park is a "research scientist" who works with microscopes and chemicals to study microorganisms.
   - The image confirms a laboratory setting with a microscope and a lab coat.
   - This aligns with the idea that Dr. Park is a scientist, not a chef, construction worker, or pilot.

Step 5: Summarize and conclude with all the information.
   - Based on the image (lab coat, microscope) and text ("research scientist studying microorganisms"), the statement most likely to be true is B.

Final answer: B

Question:

"""
    prompt = placeholder + e + text + add + hint

    return prompt


def generate_function_input_single_image(data, dataset, part):
    input = data["input"]
    image_path = f"datasets/{dataset}/images/{part}/{data['id']}.png"
    return input, image_path


def one_shot_prompt_building_single_image_completion(input, hint='', add=''):
    placeholder = f"<image>\n"
    text = input
    if hint != '':
        hint = "\n" + hint
    e = """
You have been given a question that involves both an image and a text. 
Your task is to complete the reasoning trial by following exactly five steps:

Step 1: Restate the question.
   - Clearly rephrase or clarify the question in your own words.

Step 2: Gather evidence from the image.
   - Describe any relevant visual details (e.g., objects, people, locations, interactions) that might address the question.

Step 3: Identify any background knowledge needed.
   - Note any general facts, assumptions, or external knowledge that is necessary to address the question.

Step 4: Reason with the current evidence.
   - Integrate the information from the image, text, and relevant background knowledge. 
   - Show how these pieces of evidence lead toward an answer.

Step 5: Summarize and conclude with all the information.
   - Provide a concise, direct answer to the question, referencing the supporting evidence and reasoning.

Once you have completed your reasoning, provide your final answer in the format:

Final answer: ...

Example given input:

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format \"Final answer: ..\"

We have an image showing a person wearing a white lab coat standing in front of a microscope on a lab bench.
The text states: "Dr. Park is a research scientist who works with microscopes and chemicals to study microorganisms."
Which of the following statements is most likely true about Dr. Park?
(A) Dr. Park is a chef who specializes in baking.
(B) Dr. Park is a scientist who probably conducts experiments.
(C) Dr. Park is a construction worker.
(D) Dr. Park is an airline pilot.

Step 1: Restate the question.
   - The question is asking which of the four statements is most likely true about Dr. Park, based on the provided image and text.

Step 2: Gather evidence from the image.
   - The image shows a person wearing a white lab coat, standing in front of a microscope on a lab bench.
   - This setting suggests a laboratory environment and a scientific context.

Step 3: Identify any background knowledge needed.
   - Generally, someone wearing a lab coat and using a microscope is likely engaged in scientific research or lab work.
   - Dr. Park is mentioned in the text as a researcher who studies microorganisms.
   
Your response should be:

Step 4: Reason with the current evidence.
   - The text states Dr. Park is a "research scientist" who works with microscopes and chemicals to study microorganisms.
   - The image confirms a laboratory setting with a microscope and a lab coat.
   - This aligns with the idea that Dr. Park is a scientist, not a chef, construction worker, or pilot.

Step 5: Summarize and conclude with all the information.
   - Based on the image (lab coat, microscope) and text ("research scientist studying microorganisms"), the statement most likely to be true is B.

Final answer: B

Given input:

"""
    prompt = placeholder + e + text + add + hint

    return prompt