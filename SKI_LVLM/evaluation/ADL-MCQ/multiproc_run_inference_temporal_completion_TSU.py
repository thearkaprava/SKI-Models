import torch
import os
import argparse
import json
from tqdm import tqdm
import sys
import multiprocessing as mp

import mcq_parsing_llm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_processes', required=False, default=1, type=int)
    parser.add_argument('--videochatgpt_path', help='Directory where you cloned videochatgpt', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files', required=True)
    parser.add_argument('--qa_file', help='Path to the QA file containing questions and answers', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON', required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default='video-chatgpt_v1')
    parser.add_argument("--projection_path", type=str, required=True)
    parser.add_argument("--use-token-modality-prefix", help='Use token modality prefix for the model.', action='store_true')
    parser.add_argument("--use-string-modality-prefix", help='Use string modality prefix for the model.', action='store_true')
    parser.add_argument("--model-trained-with-base-videochatgpt", help='Model youre evaluating was trained with base videochatgpt code (changes how start/end tokens are loaded).', action='store_true')
    parser.add_argument("--debug", action='store_true', help='Debug mode.')
    return parser.parse_args()

def save_to_json(output_dir, output_name, data):
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

def run_inference(process_id, args):
    available_gpus = torch.cuda.device_count()
    this_gpu = process_id % available_gpus
    torch.cuda.set_device(this_gpu)
    print(f"Process {process_id} using GPU {this_gpu}")

    #### Start of run_inference code ####
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name, args.projection_path, args.use_token_modality_prefix, args.use_string_modality_prefix, args.model_trained_with_base_videochatgpt)

    # Load QA data
    with open(args.qa_file) as file:
        qa_data = json.load(file)
    
    # Split data for multiprocessing
    read_start = process_id * len(qa_data) // args.num_processes
    read_end = len(qa_data) if process_id == args.num_processes - 1 else (process_id + 1) * len(qa_data) // args.num_processes
    qa_data = qa_data[read_start:read_end]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    conv_mode = args.conv_mode

    correct_count = 0
    total_count = 0

    if process_id == 0:
        iterator = tqdm(enumerate(qa_data), total=len(qa_data))
    else:
        iterator = enumerate(qa_data)

    processed_data = []
    for idx, sample in iterator:
        video_path = os.path.join(args.video_dir, sample['video_path'])
        question = sample['question']
        options = sample['options']
        ground_truth = sample['correct_answer']

        # Create a result dictionary for this sample
        result_dict = {
            'video_path': sample['video_path'],
            'question': question,
            'options': options,
            'ground_truth': ground_truth
        }

        # Format options string for model input
        choices_str = " ".join([f"({k}) {v}" for k, v in options.items()])
        full_question = f"{question} The output should be the choice among one of the following choices. Choices are {choices_str}"
        
        result_dict['formatted_question'] = full_question

        if os.path.exists(video_path):
            video_frames = load_video(video_path)
            if video_frames is None:
                print(f"Skipping video: {video_path}")
                continue
                
            try:
                # Get model prediction
                prediction = model_infer(video_frames, full_question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len, max_new_tokens=args.max_new_tokens)

                result_dict['prediction'] = prediction

                # Parse prediction using MCQ parsing
                prompt = mcq_parsing_llm.build_prompt(question, choices_str, prediction)
                letter_answer, llm_out = mcq_parsing_llm.parse_with_llama(prompt)
                
                result_dict['parsed_answer'] = letter_answer
                result_dict['llm_output'] = llm_out
   
                if letter_answer == ground_truth:
                    correct_count += 1
                    result_dict['is_correct'] = True
                else:
                    result_dict['is_correct'] = False

                if args.debug and process_id == 0:
                    print('================Question================')
                    print(full_question)
                    print('=================Answer=================')
                    print(f'({ground_truth})')
                    print('==================Pred==================')
                    print(f'({letter_answer}) {prediction}')
                    print('========================================')

            except Exception as e:
                print(f"Error processing video file '{video_path}': {str(e)}")
                result_dict['error'] = str(e)
        else:
            print(f"Video file not found: {video_path}")
            result_dict['error'] = 'Video file not found'
            continue

        processed_data.append(result_dict)
        total_count += 1

        if process_id == 0:
            iterator.set_description(f"(process {process_id}) Accuracy: {correct_count / total_count * 100:.2f}%")

    print(f"Final Accuracy (process {process_id}): {correct_count / total_count * 100:.2f}%")
    return (process_id, correct_count, total_count, processed_data)

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
        save_to_json(args.output_dir, f"{args.output_name}.json", result[3])
        print(f"Final Accuracy: {result[1] / result[2] * 100:.2f}%")
        exit(0)

    pool = mp.Pool(processes=args.num_processes)
    results = []
    for i in range(args.num_processes):
        print('Starting process ', i)
        result = pool.apply_async(run_inference, args=(i, args))
        results.append(result)

    pool.close()
    pool.join()

    results = [result.get() for result in results]
    results = sorted(results, key=lambda x: x[0])

    correct_count = sum(result[1] for result in results)
    total_count = sum(result[2] for result in results)
    final_accuracy = correct_count / total_count * 100

    print(f"Final Accuracy (all processes): {final_accuracy:.2f}%")
    
    # Save log file
    log_path = os.path.join(args.output_dir, f"{args.output_name}.log")
    with open(log_path, 'w') as file:
        file.write(f"Final Accuracy (all processes): {final_accuracy:.2f}%\n")
        file.write(f"Arguments: {args}\n")

    # Combine results from all processes
    all_results = []
    for result in results:
        all_results.extend(result[3])

    save_to_json(args.output_dir, f"{args.output_name}.json", all_results)