import torch
import os
import argparse
import json
from tqdm import tqdm
import re
import sys

import mcq_parsing_llm

import multiprocessing as mp
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_processes', required=False, default=1, type=int)
    parser.add_argument('--videochatgpt_path', help='Directory where you cloned videochatgpt', required=True, default='/data/users/dreilly1/Video-ChatGPT/')
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True, default='')
    parser.add_argument('--qa_file', help='Path to the QA file containing questions and answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default='llavidal_v1')
    parser.add_argument("--projection_path", type=str, required=True)
    parser.add_argument("--use-token-modality-prefix", help='Use token modality prefix for the model.', action='store_true')
    parser.add_argument("--use-string-modality-prefix", help='Use string modality prefix for the model.', action='store_true')
    parser.add_argument("--model-trained-with-base-videochatgpt", help='Model youre evaluating was trained with base videochatgpt code (changes how start/end tokens are loaded).', action='store_true')
    parser.add_argument("--max_new_tokens", type=int, default=1024, required=False, help='Maximum number of new tokens to generate.')
    parser.add_argument("--debug", action='store_true', help='Debug mode.')
    return parser.parse_args()

def save_to_json(output_dir, output_name, data):
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

def parse_options(options_str):
    options_dict = {}
    pattern = r"(\d+)\.\s*\[(.*?)\]"
    matches = re.findall(pattern, options_str)
    for match in matches:
        key, value = match
        options_dict[key] = [item.strip().strip("'") for item in value.split(',')]
    return options_dict

def run_inference(process_id, args):
    available_gpus = torch.cuda.device_count()
    this_gpu = process_id % available_gpus
    torch.cuda.set_device(this_gpu)
    print(f"Process {process_id} using GPU {this_gpu}")

    #### Start of run_inference code ####
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name, args.projection_path, args.use_token_modality_prefix, args.use_string_modality_prefix, args.model_trained_with_base_videochatgpt)

    with open(args.qa_file) as file:
        qa_data = json.load(file)
    
    read_start = process_id * len(qa_data) // args.num_processes
    if process_id == args.num_processes - 1:
        read_end = len(qa_data)
    else:
        read_end = (process_id + 1) * len(qa_data) // args.num_processes
    qa_data = qa_data[read_start:read_end]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    conv_mode = args.conv_mode

    letter_lookup = {
        '1': 'A',
        '2': 'B',
        '3': 'C',
        '4': 'D',
        '5': 'E'
    }

    correct_count = 0
    total_count = 0

    if process_id == 0:
        iterator = tqdm(enumerate(qa_data), total=len(qa_data))
    else:
        iterator = enumerate(qa_data)

    for i, sample in iterator:
        video_path = os.path.join(args.video_dir, f"{sample['id']}.mp4")
        question = sample['Q']
        options = parse_options(sample['Options']) # MCQ choices were saved as a string representation of a list in the json file, have to parse it back to a list
        ground_truth = sample['Ground Truth']

        # Ground truth is a list, have to check which MCQ choice corresponds to it
        letter_gt = None
        for k, v in options.items():
            # check if the list v is equal to the list ground_truth
            if v == ground_truth:
                letter_gt = letter_lookup[k]
                break

        assert letter_gt is not None, f"Ground truth not found in options for question: {question}"
        qa_data[i]['ground_truth_letter'] = letter_gt

        # build the choices string
        choices_str = []
        for k, v in options.items():
            letter = letter_lookup[k]
            choices_str.append(f"({letter}) {', '.join(v)}")

        choices_str = " ".join(choices_str)
        qa_data[i]['options_with_letter'] = choices_str

        full_question = f"{question} The output should be the choice among one of the following choices. Choices are {choices_str}"

        if os.path.exists(video_path):
            video_frames = load_video(video_path)
            if video_frames is None:
                print(f"Skipping video: {video_path}")
                continue
            try:
                prediction = model_infer(video_frames, full_question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len, max_new_tokens=args.max_new_tokens)
                if args.debug:
                    print('================Question================')
                    print(full_question)
                    print('=================Answer=================')
                    print(prediction)
                    print('========================================')
                qa_data[i]['prediction'] = prediction

                prompt = mcq_parsing_llm.build_prompt(question, choices_str, prediction)
                letter_answer, llm_out = mcq_parsing_llm.parse_with_llama(prompt)
                qa_data[i]['parsed_answer_from_llm'] = letter_answer

                # print(f'Question: {question}')
                # print(f'Choices: {choices_str}')
                # print(f'Prediction: {prediction}')
                # print(f'LLM Output: {llm_out}')
                # print(f'Answer: {letter_gt}')
                # print(f'Ground truth: {ground_truth}')
                # print(f'Letter Answer from LLM: {letter_answer}\n')

                if letter_answer == letter_gt:
                    correct_count += 1 
            except Exception as e:
                print(f"Error processing video file '{video_path}': {str(e)}")
        else:
            print(f"Video file not found: {video_path}")
            continue

        total_count += 1

        if process_id == 0:
            iterator.set_description(f"(process {process_id}) Accuracy: {correct_count / total_count * 100:.2f}%")

    print(f"Final Accuracy (process {process_id}): {correct_count / total_count * 100:.2f}")
    #### End of run_inference code ####

    return (process_id, correct_count, total_count, qa_data)

if __name__ == "__main__":
    args = parse_args()

    # assert args.use_token_modality_prefix or args.use_string_modality_prefix, 'You must at least one of --use-token-modality-prefix or --use-string-modality-prefix. Choose one depending on the model being evaluated.'

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

    correct_count = 0
    total_count = 0
    for result in results:
        correct_count += result[1]
        total_count += result[2]

    print(f"Final Accuracy (all processes): {correct_count / total_count * 100:.2f}%")

    # save .log file
    log_path = os.path.join(args.output_dir, f"{args.output_name}.log")
    if os.path.exists(args.output_dir):
        with open(log_path, 'a') as file:
            file.write("========================================\n")
            file.write(f"Final Accuracy (all processes): {correct_count / total_count * 100:.2f}%\n")
            file.write(f"Arguments: {args}\n")
    else:
        with open(log_path, 'w') as file:
            file.write(f"Final Accuracy (all processes): {correct_count / total_count * 100:.2f}%\n")
            file.write(f"Arguments: {args}\n")
    

    qa_data = []
    for result in results:
        qa_data.extend(result[3])

    save_to_json(args.output_dir,f"{args.output_name}.json", qa_data)

    debug = False
    if debug:
        with open(args.qa_file) as file:
            qa_data_orig = json.load(file)

        assert len(qa_data) == len(qa_data_orig)

        for sample1, sample2 in zip(qa_data, qa_data_orig):
            assert sample1['id'] == sample2['id']