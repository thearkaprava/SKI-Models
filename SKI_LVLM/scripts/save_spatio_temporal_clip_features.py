import os
import math
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import CLIPVisionModel, CLIPImageProcessor

def load_video(vis_path, num_frm=100):
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)
    total_num_frm = min(total_frame_num, num_frm)
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    img_array = vr.get_batch(frame_idx).asnumpy()
    a, H, W, _ = img_array.shape
    h, w = 224, 224
    if img_array.shape[-3] != h or img_array.shape[-2] != w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
    clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(total_num_frm)]
    return clip_imgs

def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = [(int(np.round(seg_size * i)) + int(np.round(seg_size * (i + 1)))) // 2 for i in range(desired_num_frames)]
    return seq

def get_spatio_temporal_features(features, num_temporal_tokens=100):
    t, s, c = features.shape
    temporal_tokens = np.mean(features, axis=1)
    padding_size = num_temporal_tokens - t
    if padding_size > 0:
        temporal_tokens = np.pad(temporal_tokens, ((0, padding_size), (0, 0)), mode='constant')
    spatial_tokens = np.mean(features, axis=0)
    sp_features = np.concatenate([temporal_tokens, spatial_tokens], axis=0)
    return sp_features

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--video_dir_path", required=False, help="Path to read the videos from",default="")
    parser.add_argument("--clip_feat_path", required=False, help="The output dir to save the features in",default="")
    parser.add_argument("--infer_batch", required=False, type=int, default=48, help="Number of frames/images to perform batch inference.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    video_dir_path = args.video_dir_path
    clip_feat_path = args.clip_feat_path
    infer_batch = args.infer_batch
    os.makedirs(clip_feat_path, exist_ok=True)

    # Initialize the CLIP model
    image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16)
    vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
    vision_tower.eval()

    for subdir, dirs, files in os.walk(video_dir_path):
        for file in tqdm(files):
            if file.endswith(".mp4"):
                video_path = os.path.join(subdir, file)
                video_id = file.split('.')[0]
                if os.path.exists(f"{clip_feat_path}/{video_id}.pkl"):
                    continue
                try:
                    video = load_video(video_path)
                    video_tensor = image_processor.preprocess(video, return_tensors='pt')['pixel_values']
                    video_tensor = video_tensor.half()
                    n_chunk = len(video_tensor)
                    video_features = torch.FloatTensor(n_chunk, 256, 1024).fill_(0)
                    n_iter = int(math.ceil(n_chunk / float(infer_batch)))
                    for i in range(n_iter):
                        min_ind = i * infer_batch
                        max_ind = (i + 1) * infer_batch
                        video_batch = video_tensor[min_ind:max_ind].cuda()
                        image_forward_outs = vision_tower(video_batch, output_hidden_states=True)
                        select_hidden_state_layer = -2
                        select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
                        batch_features = select_hidden_state[:, 1:].detach().cpu()
                        video_features[min_ind:max_ind] = batch_features
                    with open(f"{clip_feat_path}/{video_id}.pkl", 'wb') as f:
                        pickle.dump(get_spatio_temporal_features(video_features.numpy().astype("float16")), f)
                except Exception as e:
                    print(f"Can't process {video_path}: {str(e)}")

if __name__ == "__main__":
    main()








