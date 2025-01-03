import torch

import argparse
import json
import time
import os

import multiprocessing as mp
from tqdm import tqdm

import sys


def parse_args():
    """
    Parse command-line arguments.s
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--num_processes', required=False, default=1, type=int)
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


def run_inference(process_id, args):
    available_gpus = torch.cuda.device_count()
    this_gpu = process_id % available_gpus
    torch.cuda.set_device(this_gpu)
    print(f"Process {process_id} using GPU {this_gpu}")

    #### Start of run_inference code ####
    # Initialize the model
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name, args.projection_path, args.use_token_modality_prefix, args.use_string_modality_prefix, args.using_base_videochatgpt_weights)

    # Load the ground truth file
    with open(args.gt_file) as file:
        qa_data = json.load(file)
    
    read_start = process_id * len(qa_data) // args.num_processes
    if process_id == args.num_processes - 1:
        read_end = len(qa_data)
    else:
        read_end = (process_id + 1) * len(qa_data) // args.num_processes
    qa_data = qa_data[read_start:read_end]

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    conv_mode = args.conv_mode

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Dominick: Why is this not used?
    fixed_question = "Please describe the primary actions and interactions in the video, focusing on movements and the use of objects by any person or persons present. Describe the sequence of events in detail but avoid mentioning clothing or background elements unless they are integral to the action being performed. The description should succinctly encapsulate the essence of the activity within the scene, aiming for a concise depiction in no more than 100 words. Focus on how the objects are being interacted with and any significant changes they undergo as a result of the actions taken. Describe the actions in details."

    if process_id == 0:
        iterator = tqdm(enumerate(qa_data), total=len(qa_data))
    else:
        iterator = enumerate(qa_data)

    for i, sample in iterator:

        video_name = sample['video_name']
        question = sample['Q']
        gt_answer = sample['A']

        # Load the video file
        for fmt in video_formats:
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        # Check if the video exists
        if video_path is not None:
            video_frames = load_video(video_path)

        try:
            # Run inference on the video and add the output to the list
            start_time = time.time()
            output = model_infer(video_frames, question, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len)
            forward_time = time.time() - start_time

            qa_data[i]['pred'] = output

            if args.debug or forward_time > 10:
                if not args.debug and forward_time > 10:
                    print(f"Warning: Inference time for video {video_name} took {forward_time:.2f} seconds. Probably gibberish outputs. Printing below")
                print('================Question================')
                print(question)
                print('===============Pred Answer==============')
                print(output)
                print('===============True Answer==============')
                print(gt_answer)
                print('========================================')

        except Exception as e:
            print(f"Error processing video file '{video_name}': {e}")
    #### End of run_inference code ####

    return (process_id, qa_data)


if __name__ == "__main__":
    args = parse_args()

    sys.path.append(args.videochatgpt_path)

    try:
        from video_chatgpt.eval.model_utils import initialize_model, load_video
        from video_chatgpt.inference import video_chatgpt_infer as model_infer
    except ImportError as e:
        from llavidal.eval.model_utils import initialize_model, load_video
        from llavidal.inference import llavidal_infer as model_infer
    
    if args.num_processes == 1:
        result = run_inference(0, args)
        assert False

    pool = mp.Pool(processes=args.num_processes)

    results = []
    for i in range(args.num_processes):
        print('Starting process ', i)
        result = pool.apply_async(run_inference, args=(i,args))
        results.append(result)

    pool.close()
    pool.join()

    # parse multiprocessing results
    results = [result.get() for result in results]
    results = sorted(results, key=lambda x: x[0])

    qa_data = []
    for result in results:
        qa_data.extend(result[1])

    # Save the output list to a JSON file
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(qa_data, file, indent=4)