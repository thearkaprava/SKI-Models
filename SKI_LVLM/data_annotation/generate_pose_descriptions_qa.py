import openai
import os
import json
import time
import ast
import argparse
import warnings
from tqdm import tqdm
from pathlib import Path
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

def annotate(args, video_ids, actions_list, pose_description):
    """
    Generate question-answer pairs using actions list and pose description
    summarized from off-the-shelf models using OpenAI GPT-3.
    """
    openai.api_key = args.api_key  # Set the API key directly

    start_time = time.time()
    results = []

    for video_id in tqdm(video_ids, desc="Processing videos"):
        try:
            #actions = next((item["actions"] for item in actions_list if item["video_id"] == video_id), None)
            video_prefix = video_id[:3]
            actions= actions_list[video_id]
           
            poses = pose_description[video_prefix][video_id]["pose_desc"]

            if actions is None:
                print(f"No actions found for video ID: {video_id}")
                continue

            # Generate QA pairs with OpenAI GPT-3: Motion Analysis
            completion_0 = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You play two roles: a human asking questions related to analyzing movements and motions in a video and an intelligent chatbot designed for video analysis. "
                            "Your task is motion analysis. "
                            "As an AI assistant, assume that you have watched the video and generated the provided actions list and pose description. "
                            "Your task is to play the role of a human who asks two questions related to analyzing the movements and motions in the video and then play the role of an AI assistant that provides answers based on the actions list and pose description."
                            "------"
                            "##TASK:"
                            "Users will provide an actions list JSON describing the actions performed in the video and a pose description JSON describing how the joints and parts of the body like head, hands, etc. are moving. You will generate a set of two conversation-like questions related to analyzing the movements and motions in the video. "
                            "The questions and answers should focus on understanding the relationship between the described joint movements and the actions performed, as well as how the motion analysis describes the individual's movements. "
                            "The answers should be based on the provided actions list and pose description. "
                            "You have information about the video based on the actions list and pose description and can analyze the movements and motions accordingly."
                            "Generate TWO different questions asking about the movements and motions in the video and provide answers to each based on the actions list and pose description. "
                            "------"
                            "##INSTRUCTIONS:"
                            "- The questions must be like a human conversation and focused on analyzing the movements and motions in the video. "
                            "- The answers must be based on the provided actions list and pose description, and they should relate the joint movements to the actions performed. "
                            "------"
                            "##SAMPLE QUESTIONS:"
                            "- How do the described joint movements relate to the actions performed in the video?"
                            "- What are the movements of the joints in the video?"
                    },
                    {
                        "role": "user",
                        "content":
                            f"The actions list is: {actions}. "
                            f"The pose description is: {poses}"
                            "Generate two different questions on analyzing the movements and motions in the video, and provide answers based on the given actions list and pose description. "
                            "Please attempt to form question and answer pairs based on the two sets of information."
                            '''Please generate the response in the form of a Python list of dictionary string with keys "Q" for question and "A" for answer. Each corresponding value should be the question and answer text respectively. '''
                            '''For example, your response should look like this: [{"Q": "Your first question here...", "A": "Your first answer here..."}, {"Q": "Your first question here...", "A": "Your first answer here..."}]. '''
                            "Emphasize that the questions and answers should focus on understanding the relationship between the described joint movements and the actions performed, there should be ONE question that should discuss the joint movements related to the actions performed in the video."
                    }
                ]
            )
            try:
                # Extract Motion Analysis Based QA pairs
                response_message_0 = completion_0.choices[0].message.content
                response_message_0 = enforce_closing_bracket(response_message_0)
                response_dict_0 = ast.literal_eval(response_message_0)
            except AttributeError as e:
                print(f"Error accessing completion data: {e}")

            if response_dict_0 is not None:
                response_dict_1 = {"id": video_id}
                response_dict_0.append(response_dict_1)
                results.append(response_dict_0)

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
    parser.add_argument("--actions_list_file", required=True, help="Path to the actions list JSON file.")
    parser.add_argument("--pose_description_file", required=True, help="Path to the pose description JSON file.")
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

    # Read actions list file
    with open(args.actions_list_file) as f:
        actions_list = json.load(f)

    # Read pose description file
    with open(args.pose_description_file) as f:
        pose_description = json.load(f)

    # Get the video_ids from the pose description
    video_ids = [video_id for outer_dict in pose_description.values() for video_id in outer_dict.keys()]

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

    task_args = [(args, part, actions_list, pose_description) for part in all_parts]

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