# Required Libraries
import openai
import os
import json
import time
import ast
import argparse
import warnings
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from multiprocessing.pool import Pool

# Suppressing all warnings
warnings.filterwarnings('ignore')

def enforce_closing_bracket(text):
    # Add a closing bracket to the text if it's missing
    if text.strip()[-2:] != '}]':
        if text.strip()[-3:] == '}\n]':
            return text
        else:
            modified_text = text.replace(']', '}]')
            return modified_text
    if text.strip()[-3] != '}\n]':
        modified_text = text.replace('\n,]', '}\n]')
        return modified_text
    else:
        return text

def safe_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except SyntaxError:
        # Handle the unmatched '}' by returning None or an appropriate placeholder
        print("Syntax error encountered in response. Skipping.")
        return None    

def annotate(args, video_ids):
    """
    Generate question-answer pairs using caption and
    dense-captions summarized from off-the-shelf models using OpenAI GPT-3.
    """
    client = OpenAI(api_key=args.api_key)  # Create the OpenAI client object inside the function

    with open(args.cogvlm_captions_file) as f:
        cogvlm_captions = json.load(f)

    with open(args.detection_captions_file) as f:
        detection_captions = json.load(f)
    start_time = time.time()
    results = []
    for video_id in tqdm(video_ids, desc="Processing videos"):
        try:
            caption = cogvlm_captions[video_id]['A']
            detections = detection_captions[video_id]

            mega_caption = ""

            for detection in detections:
                mega_caption += detection

            while(True):
                # Generate QA pairs with OpenAI GPT-3: Summarization
                completion_0 = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content":
                                "You play two roles: a human asking questions related to summarizing a video and an intelligent chatbot designed for video summarization and dense captioning. "
                                "Your task is video summarization. "
                                "As an AI assistant, assume that you have watched the video and generated the provided caption as the summary of the video. "
                                "Your task is to play the role of a human who asks three questions related to summarizing the video and then play the role of an AI assistant that provides paraphrased answers based on the video content and the provided caption."
                                "------"
                                "##TASK:"
                                "Users will provide a caption of the video alongside dense caption describing detected objects in that scene, and you will generate a set of three conversation-like questions related to summarizing the video. "
                                "The questions and answers can be very similar, but they should all focus on summarizing the video content. "
                                "The answers should be paraphrased versions of the provided caption and the dense caption with the object detections. "
                                "You have information about the video based on the provided caption and have summarized the events in it. You also have the dense caption with the object and scene details."
                                "Generate THREE different questions asking to summarize the video and provide detailed answers to each based on the caption and the dense caption. "
                                "------"
                                "##INSTRUCTIONS:"
                                "- The questions must be like a human conversation and focused on summarizing the video. "
                                "- The answers must be paraphrased versions of the provided caption and the dense caption, and they should be detailed and descriptive. "
                                "------"
                                "##SAMPLE QUESTIONS:"
                                "- Can you provide a summary of the video?"
                                "- What are the main events in the video?"
                                "- Could you briefly describe the video content?"
                        },
                        {
                            "role": "user",
                            "content":
                                f"The video caption is: {caption}. "
                                f"The additional dense caption is: {mega_caption}"
                                "Generate three different questions on summarizing the video, and provide answers that are paraphrased versions of the given caption and the dense caption. "
                                "Please attempt to form question and answer pairs based on the two sets of text."
                                '''Please generate the response in the form of a Python list of dictionary string with keys "Q" for question and "A" for answer. Each corresponding value should be the question and answer text respectively. '''
                                '''For example, your response should look like this: [{"Q": "Your first question here...", "A": "Your first answer here..."}, {"Q": "Your first question here...", "A": "Your first answer here..."}, {"Q": "Your first question here...", "A": "Your first answer here..."}]. '''
                                "Emphasize that the questions and answers can be very similar, but they should all focus on summarizing the video content."
                        }
                    ]
                )
                try:
                    # Extract Summary Based QA pairs
                    # Convert response to a list of dictionary.
                    response_message_0 = completion_0.choices[0].message.content
                    response_message_0 = enforce_closing_bracket(response_message_0)
                    response_dict_0 = ast.literal_eval(response_message_0)
                    #response_dict_0 = safe_literal_eval(response_message_0)
                except AttributeError as e:
                    print(f"Error accessing completion data: {e}")
            

                # Generate QA pairs with OpenAI GPT-3: Caption Based
                # Answers specifically restricted to information in the caption
                completion_1 = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content":
                                "You play two roles: a human asking questions related a video and an intelligent chatbot designed for video summarization and dense captioning. "
                                "Your task is extracting diverse video information. "
                                "As an AI assistant, assume that you have watched the video and generated the provided caption as the summary of the video. "
                                "Your task is to play the role of a human who asks three questions related to summarizing the video and then play the role of an AI assistant that provides paraphrased answers based on the video content and the provided caption."
                                "------"
                                "##TASK:"
                                "Users will provide a caption of the video alongside dense caption describing detected objects,setting and details in that scene, and you will generate a set of three conversation-like questions related to the video. "
                                "The questions and answers can be very similar, but they should all focus on the details of the video content. "
                                "The answers should be paraphrased versions of the provided caption and the dense caption with the object and scene details. "
                                "You have information about the video based on the provided caption and have summarized the actions in it. You also have the dense caption with the scene details."
                                "Generate THREE different questions asking the details of the video and provide detailed answers to each based on the caption and the dense caption and one question should be about what actions are happening which should come from captions of the video. "
                                "------"
                                "##INSTRUCTIONS:"
                                "- The questions must be like a human conversation and focused on finding the intricate and unique details of the video. "
                                "- The answers must be paraphrased versions of the provided caption and the dense caption, and they should be detailed and descriptive. "
                                "------"
                                "##SAMPLE QUESTIONS:"
                                "- What are the actions occuring sequentially in the video?"
                                "- What are the colors of the outfits of the person in the video? "
                                "- What are the objects in the scene?"
                                "- What is person doing?"
                        },
                        {
                            "role": "user",
                            "content":
                                f"The video caption is: {caption}. "
                                f"The additional dense caption is: {mega_caption}"
                                "Generate three different questions on the details of the video, and provide answers that are paraphrased versions of the given caption and the dense caption. "
                                "Please attempt to form question and answer pairs based on the two sets of text."
                                '''Please generate the response in the form of a Python list of dictionary string with keys "Q" for question and "A" for answer. Each corresponding value should be the question and answer text respectively. '''
                                '''For example, your response should look like this: [{"Q": "Your first question here...", "A": "Your first answer here..."}, {"Q": "Your first question here...", "A": "Your first answer here..."}, {"Q": "Your first question here...", "A": "Your first answer here..."}]. '''
                                "Emphasize that the questions and answers can be very similar, but they should all focus on the various details of the video content and understanding what actions are happening."
                                "Include at least one question about the sequence of actions happening in the video."
                        }
                    ]
                )
                try:
                    response_message_1 = completion_1.choices[0].message.content  # Adjust this line based on the actual API
                    response_message_1 = enforce_closing_bracket(response_message_1)
                    response_dict_1 = safe_literal_eval(response_message_1)        # Save the response dictionary into a JSON file
                    if response_dict_1 is not None:
                        combined_responses = response_dict_0 + response_dict_1
                        response_dict_3 = {"id":video_id}
                        combined_responses.append(response_dict_3)
                        results.append(combined_responses)
                        break

                except AttributeError as e:
                    print(f"Error accessing completion data: {e}")

        except Exception as e:
            print(f"Error processing video_id {video_id}: {e}")
            print("Skipping video_id and proceeding to the next one.")
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds -----> {elapsed_time/60:.2f} minutes") 
    return results
def parse_args():
    """
    Command-line argument parser.
    """
    parser = argparse.ArgumentParser(description="Descriptive question-answer-generation-using-GPT-3")
    parser.add_argument("--cogvlm_captions_file", required=True, help="Path to the CogVLM captions JSON file.")
    parser.add_argument("--detection_captions_file", required=True, help="Path to the detection captions JSON file.")
    parser.add_argument("--output_dir", required=True, help="Path to save the annotation JSON files.")
    parser.add_argument("--api_key", required=True, help="OpenAI API key.")
    parser.add_argument("--num_tasks", required=False, type=int, help="Number of splits.", default=10)

    return parser.parse_args()
def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments
    args = parse_args()

    # Create output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Read CogVLM captions file
    with open(args.cogvlm_captions_file) as f:
        cogvlm_captions = json.load(f)

    # Get the video_ids
    video_ids = list(cogvlm_captions.keys())

    num_tasks = args.num_tasks

    # Files that have already been completed.
    completed_files = os.listdir(args.output_dir)
    print(f"completed_files: {len(completed_files)}")

    # Files that have not been processed yet.
    incomplete_video_ids = [video_id for video_id in video_ids if f"{video_id}.json" not in completed_files]
    print(f"incomplete_video_ids: {len(incomplete_video_ids)}")

    if len(incomplete_video_ids) == 0:
        print("All tasks completed!")
        return

    # Split tasks into parts.
    num_tasks = min(len(incomplete_video_ids), num_tasks)
    part_len = len(incomplete_video_ids) // num_tasks
    all_parts = [incomplete_video_ids[i:i + part_len] for i in range(0, len(incomplete_video_ids), part_len)]

    task_args = [(args, part) for part in all_parts]

    # Use a pool of workers to process the video_ids in parallel.
    with Pool() as pool:
        results = pool.starmap(annotate, task_args)

    # Flatten the results list
    flattened_results = [item for sublist in results for item in sublist]

    # Save the results to a JSON file
    output_path = os.path.join(args.output_dir, "results.json")
    with open(output_path, "a") as f:
        json.dump(flattened_results, f, indent=4)
        print(f"Completed, Annotations saved in {output_path}")

  
if __name__ == "__main__":
    main()



