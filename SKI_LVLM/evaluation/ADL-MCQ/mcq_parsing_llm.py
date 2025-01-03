import ollama
import openai
import re


def build_prompt(question, options, prediction):
    prompt = f'''
You are an AI agent designed to evaluate responses to multiple-choice questions. Your task is to determine the best answer to the following question based on the provided options. You should not explain your answer or provide any additional information, you should only provide the letter of the multiple choice answer that is most similar to the free-form response given to you, start your reply with "The best answer is". If the free-form response is different from all of the provided answers, the letter of the answer should be 'Z'. Here are some examples of your task:

Example 1:
Question: What is the main object in image?
Options: (A) teddy bear (B) rabbit (C) cat (D) dog
Answer: the main object in the image is a cute teddy bear
The best answer is: (A)

Example 2:
Question: What action is being performed by the person in the video?
Options: (A) walking (B) cleaning up (C) drinking from bottle (D) drinking from cup
Answer: the person in the video is seen in the kitchen, holding a cup and drinking from it
The best answer is: (C)

Example 3:
Question: What action will the person perform next?
Options: (A) sit down (B) stand up (C) walk to the door (D) walk to the window
Answer: the person is in the kitchen and holding a cup, they will most likely walk to the fridge to get some milk
The best answer is: (Z)

Example 4:
Question: Given that the person in the video "walked to the kitchen" and then "picked up the dirty plate", what action are they most likely to perform next?
Options: (A) leave the kitchen (B) walk to the sink (C) start cooking eggs (D) turn on the stove
Answer: seeing as how the person walked to the kitchen and then picked up the dirty plate, the most likely action they will perform next is to walk to the sink to wash the plate. This is because there is a sink behind them and their intention seems to be to clean the plate.
The best answer is: (B)
-----------------------------------------

Your task is to evaluate the following question and provide the best answer based on the options provided. You should not provide any additional information or explanation, only the letter of the multiple choice answer that is most similar to the free-form response given to you.
Question: {question}
Options: {options}
Answer: {prediction}
Please complete the following: The best answer is:
    '''
    return prompt

def extract_characters_regex(s, choices=['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(Z)']):
    if type(s) is dict:
        s = ''
    s = s.strip()
    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is'
        'The correct option is',
        'Best answer:'
        'Best option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[ABCDE]', s):
        return ''
    matches = re.search(r'[ABCDE]', s)
    if matches is None:
        for choice in choices:
            if s.lower() in choice.lower():
                return choice[1]
        return ''
    return matches[0]


def parse_with_llama(prompt):
    prompt_to_llm = [
        {
            'role': 'user',
            'content': prompt
        }
    ]

    response = ollama.chat(model='llama3.1',  messages=prompt_to_llm)
    response = response['message']['content']

    answer_letter = extract_characters_regex(response)

    return answer_letter, response

def parse_with_chatgpt(prompt, model='gpt-3.5-turbo', api_key=None):
    assert api_key is not None, 'Please provide an OpenAI API key.'
    openai.api_key = api_key

    prompt_to_llm = [
        {
            'role': 'user',
            'content': prompt
        }
    ]

    response = openai.ChatCompletion.create(model=model, messages=prompt_to_llm)
    response = response['choices'][0]['message']['content']

    answer_letter = extract_characters_regex(response)

    return answer_letter, response