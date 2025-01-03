import os
import argparse
import json
from tqdm import tqdm

import sys


def parse_args():
    """
    Parse command-line arguments.s
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--videochatgpt-path', help='Directory where you cloned videochatgpt', required=True, default='/data/users/dreilly1/Video-ChatGPT/')
    parser.add_argument('--video_dir', help='Directory containing video files.', required=False,default='/data/CHARADES/Charades_v1_480')
    parser.add_argument('--gt_file', help='Path to the ground truth file.', required=False,default='/data/CHARADES/test_set_captions.json')
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=False,default='/home/rchakra6/llavidal/ADLX_cropped_videos')
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=False,default='charades_ADLX_cropped_videos')
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, required=False, default='llavidal_v1')
    parser.add_argument("--projection_path", type=str, required=True,default="/home/rchakra6/llavidal/ADLX_cropped_videos.bin")
    parser.add_argument('--use-token-modality-prefix', help='Use token modality prefix.', action='store_true')
    parser.add_argument('--use-string-modality-prefix', help='Use string modality prefix.', action='store_true')
    parser.add_argument('--using-base-videochatgpt-weights', help='If passing base videochatgpt weights.', action='store_true')
    parser.add_argument('--debug', help='Debug mode.', action='store_true')

    return parser.parse_args()


def run_inference(args):
    """
    Run inference on a set of video files using the provided model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name, args.projection_path, args.use_token_modality_prefix, args.use_string_modality_prefix, args.using_base_videochatgpt_weights)

    # Load the ground truth file
    with open(args.gt_file) as file:
        gt_contents = json.load(file)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results
    conv_mode = args.conv_mode

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Define the fixed question here
    fixed_question = "Please describe the primary actions and interactions in the video, focusing on movements and the use of objects by any person or persons present. Describe the sequence of events in detail but avoid mentioning clothing or background elements unless they are integral to the action being performed. The description should succinctly encapsulate the essence of the activity within the scene, aiming for a concise depiction in no more than 100 words. Focus on how the objects are being interacted with and any significant changes they undergo as a result of the actions taken. Describe the actions in details."

    for sample in tqdm(gt_contents):
        video_name = sample['video_name']
        sample_set = sample
        question = sample['Q']
        gt_answer = sample['A']

        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        # Check if the video exists
        if video_path is not None:  # Modified this line
            video_frames = load_video(video_path)

        try:
            # Run inference on the video and add the output to the list
            output = model_infer(video_frames, question, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len)
            sample_set['pred'] = output

            if args.debug:
                print('================Question================')
                print(question)
                print('===============Pred Answer==============')
                print(output)
                print('===============True Answer==============')
                print(gt_answer)
                print('========================================')

            output_list.append(sample_set)
        except Exception as e:
            print(f"Error processing video file '{video_name}': {e}")

    # Save the output list to a JSON file
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()

    sys.path.append(args.videochatgpt_path)

    try:
        from video_chatgpt.eval.model_utils import initialize_model, load_video
        from video_chatgpt.inference import video_chatgpt_infer as model_infer
    except ImportError as e:
        from llavidal.eval.model_utils import initialize_model, load_video
        from llavidal.inference import llavidal_infer as model_infer

    run_inference(args)