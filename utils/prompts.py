CoT_prompts_zero_shot = """
You have been given information from an image and a text. Your task is to answer the question below by repeatedly analyzing the image and the text, step by step. For each step:

1. If you are analyzing the image, begin with:
   Step n: Analyze the visual details in the image. Describe objects, actions, and context observeD.

2. If you are analyzing the text, begin with:
   Step n: Examine the textual content. Note key details, context, or relationships mentioned.

3. If you are summarizing or synthesizing, begin with:
   Step n: Summarize and integrate the details gathered so far from the image and text. Identify any relevant connections.

4. If needed, refine your reasoning further:
   Step n: Revisit earlier steps to check for new insights or contradictions in the combined information.

Repeat these steps as many times as necessary to capture all relevant details. Each step should build on the information from previous steps.

Once you have completed your reasoning, provide your final answer in the format:

Final answer: ...

Question:
"""

CoT_prompts_one_shot = """
You have been given information from an image and a text. Your task is to answer the question below by repeatedly analyzing the image and the text, step by step. For each step:

1. If you are analyzing the image, begin with:
   Step n: Analyze the visual details in the image. Describe objects, actions, and context you observe.

2. If you are analyzing the text, begin with:
   Step n: Examine the textual content. Note key details, context, or relationships mentioned.

3. If you are summarizing or synthesizing, begin with:
   Step n: Summarize and integrate the details gathered so far from the image and text. Identify any relevant connections.

4. If needed, refine your reasoning further:
   Step n: Revisit earlier steps to check for new insights or contradictions in the combined information.

Repeat these steps as many times as necessary to capture all relevant details. Each step should build on the information from previous steps.

Once you have completed your reasoning, provide your final answer in the format:

Final answer: ...

Example:

Question:
We have an image showing a small dog sitting beside a large doghouse. The text says: 
'The owner bought this doghouse thinking it was the right size, but clearly, it is too big for his new puppy.'
Is the doghouse appropriately sized for the puppy or not?

Step 1: Analyze the visual details in the image. The image shows a small dog (looks like a puppy) sitting beside a much larger doghouse. The doghouse appears to be big compared to the size of the puppy.

Step 2: Examine the textual content. The text states: “The owner bought this doghouse thinking it was the right size, but clearly, it is too big for his new puppy.”

Step 3: Summarize and integrate the details gathered so far from the image and text. Visually, the doghouse dwarfs the puppy. Textually, it’s mentioned that the doghouse is too big, confirming the visual observation.

Step 4: Revisit earlier steps to check for new insights or contradictions. No contradictions: both the image and text suggest that the doghouse is indeed larger than necessary for the puppy.

Final answer: 
No, the doghouse is not appropriately sized for the puppy; it is clearly too big.
"""

CoT_prompts_zero_shot_2 = """
You have been given a question that involves both an image and a text. 
Your task is to answer the question step by step. 
You should answer by following these steps exactly once:

Step 1: Restate the question.
   - Briefly restate or clarify the question in your own words.

Step 2: Gather evidence from the image.
   - Describe any relevant visual details that might address the question.

Step 3: Reason with the current evidence.
   - Interpret the image evidence in relation to the question.

Step 4: Gather evidence from the text.
   - Identify key points, facts, or context from the text that might answer the question.

Step 5: Reason with the combined evidence.
   - Integrate the information from both the image and the text. Check if they confirm, contradict, or supplement each other.

Once you have completed your reasoning, provide your final answer in the format:

Final answer: ...

Question:
"""

CoT_prompts_one_shot_2 = """
You have been given a question that involves both an image and a text. 
Your task is to answer the question step by step. 
You should answer by following these steps exactly once:

Step 1: Restate the question.
   - Briefly restate or clarify the question in your own words.

Step 2: Gather evidence from the image.
   - Describe any relevant visual details that might address the question.

Step 3: Reason with the current evidence.
   - Interpret the image evidence in relation to the question.

Step 4: Gather evidence from the text.
   - Identify key points, facts, or context from the text that might answer the question.

Step 5: Reason with the combined evidence.
   - Integrate the information from both the image and the text. Check if they confirm, contradict, or supplement each other.

Once you have completed your reasoning, provide your final answer in the format:

Final answer: ...

Example:

Question:
There is an image showing a child hugging a large brown dog.
This child just adopted a new pet dog named Bruno. The dog is very friendly and loves playing fetch. Is the child's new pet named Bruno?


Choices:
(A) Yes
(B) No

Step 1: Restate the question.
   - We want to determine whether the child's new pet is named Bruno.

Step 2: Gather evidence from the image.
   - The image shows a child hugging a large brown dog. 
   - It appears the dog and the child are friendly and comfortable with each other.

Step 3: Reason with the current evidence.
   - From the image alone, we see a friendly interaction, but there’s no direct textual label of the dog's name in the image.
   - The dog is brown and large, which may align with "Bruno" mentioned in the text, but let's confirm via the text.

Step 4: Gather evidence from the text.
   - The text states: "This child just adopted a new pet dog named Bruno. The dog is very friendly and loves playing fetch."

Step 5: Reason with the combined evidence.
   - Both the text and the image consistently show a friendly dog with a child.
   - The text explicitly says the new pet dog is named Bruno. 
   - The image does not contradict this (it is indeed a large brown dog).

Final answer: A

Question:
"""

CoT_prompts_zero_shot_3 = """
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

Question:
"""


CoT_prompts_one_shot_3 = """
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
"""

