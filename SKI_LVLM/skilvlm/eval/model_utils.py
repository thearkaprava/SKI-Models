
from skilvlm.model import SKILVLMLlamaForCausalLM
from skilvlm.utils import disable_torch_init
from skilvlm.constants import *
import os
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
import torch
from decord._ffi.base import DECORDError



def load_video(vis_path, n_clips=1, num_frm=100):
    """
    Load video frames from a video file.
    Parameters:
    vis_path (str): Path to the video file.
    n_clips (int): Number of clips to extract from the video. Defaults to 1.
    num_frm (int): Number of frames to extract from each clip. Defaults to 100.
    Returns:
    list: List of PIL.Image.Image objects representing video frames, or None if an error occurs.
    """
    try:
        # Load video with VideoReader
        vr = VideoReader(vis_path, ctx=cpu(0))
        total_frame_num = len(vr)
        # Currently, this function supports only 1 clip
        assert n_clips == 1
        # Calculate total number of frames to extract
        total_num_frm = min(total_frame_num, num_frm)
        # Get indices of frames to extract
        frame_idx = get_seq_frames(total_frame_num, total_num_frm)
        # Extract frames as numpy array
        img_array = vr.get_batch(frame_idx).asnumpy()
        # Set target image height and width
        target_h, target_w = 224, 224
        # If image shape is not as target, resize it
        if img_array.shape[-3] != target_h or img_array.shape[-2] != target_w:
            img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
            img_array = torch.nn.functional.interpolate(img_array, size=(target_h, target_w))
            img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
        # Reshape array to match number of clips and frames
        img_array = img_array.reshape(
            (n_clips, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
        # Convert numpy arrays to PIL Image objects
        clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(total_num_frm)]
        return clip_imgs
    except DECORDError as e:
        print(f"Error loading video: {vis_path}")
        print(f"Error message: {str(e)}")
        return None


def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return seq


def initialize_model(model_name, projection_path=None, use_token_modality_prefix=True, use_string_modality_prefix=False, using_base_videochatgpt_weights=False):
    """
    Initializes the model with given parameters.

    Parameters:
    model_name (str): Name of the model to initialize.
    projection_path (str, optional): Path to the projection weights. Defaults to None.
    use_token_modality_prefix (bool, optional): Whether to use token modality prefix (e.g., <video_start><video_end>). Defaults to True.
    use_string_modality_prefix (bool, optional): Whether to use string modality prefix (e.g., ' video: '). Defaults to False.

    Returns:
    tuple: Model, vision tower, tokenizer, image processor, vision config, and video token length.
    """

    # Disable initial torch operations
    disable_torch_init()

    # Convert model name to user path
    model_name = os.path.expanduser(model_name)

    # Load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained('mmaaz60/LLaVA-7B-Lightening-v1-1')

    # Load model
    model = SKILVLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                         use_cache=True)
    
    hidden_size_video_encoder = 1024
    model.model.mm_projector = torch.nn.Linear(hidden_size_video_encoder, model.config.hidden_size).to(model.dtype)

    # Load image processor
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

    # Set to use start and end tokens for video
    mm_use_vid_start_end = use_token_modality_prefix

    '''
    Add modality tokens to the tokenizer.

    The projection weights will
        (1) always have 3 modality patch tokens (video, object, pose) and
        (2) maybe have 6 modality start/end tokens (video, object, pose) if mm_use_vid_start_end was True when training the model

    Because we only use video modality at inference, we only add video modality tokens here. So we have to filter the video modality tokens from the projection weights.
    '''
    ## Add the patch tokens (these are always used)
    if not using_base_videochatgpt_weights:
        modality_patch_tokens = [DEFAULT_VIDEO_PATCH_TOKEN, DEFAULT_OBJECT_PATCH_TOKEN, DEFAULT_POSE_PATCH_TOKEN] # token id will be 32004
    else:
        modality_patch_tokens = [DEFAULT_VIDEO_PATCH_TOKEN]

    tokenizer.add_tokens(modality_patch_tokens, special_tokens=True)

    ## Add the start and end tokens for video, object, and pose
    if mm_use_vid_start_end:
        if not using_base_videochatgpt_weights:
            modality_prefix_tokens = [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, DEFAULT_OBJECT_START_TOKEN, DEFAULT_OBJECT_END_TOKEN, DEFAULT_POSE_START_TOKEN, DEFAULT_POSE_END_TOKEN] # token id will be 32005, 32006
        else:
            modality_prefix_tokens = [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN]

        tokenizer.add_tokens(modality_prefix_tokens, special_tokens=True)

    # Resize token embeddings of the model
    model.resize_token_embeddings(len(tokenizer)) # will be 32004 or 32006 (if mm_use_vid_start_end is True)


    # Load the weights from projection_path after resizing the token_embeddings
    if projection_path:
        print(f"Loading weights from {projection_path}")
        projector_weights = torch.load(projection_path, map_location='cpu')
        print(f'All keys of weights to load projector_weights: {projector_weights.keys()}')

        # lm_head weights are not expected in the projection weights
        if any('lm_head' in key for key in projector_weights.keys()):
            raise ValueError("lm_head weights are not expected in the projection weights. If you want to load lm_head comment out this line")
        
        if 'model.embed_tokens.weight' in projector_weights:
            print(f'Vocab size & embedding size of loaded model (i.e., model.embed_tokens.weight of projector_weights): {projector_weights["model.embed_tokens.weight"].shape}')

            vocab_size_of_proj = projector_weights['model.embed_tokens.weight'].shape[0]

            if using_base_videochatgpt_weights:
                assert vocab_size_of_proj == 32006, "Vocab size of model trained using base videochatgpt code should be 32006"

            if model.get_model().embed_tokens.weight.shape[0] == vocab_size_of_proj:
                # Vocab size of model for inference matches the vocab size of the model being loaded. So we can load the weights as is.
                pass
            elif vocab_size_of_proj == 32006 and mm_use_vid_start_end:
                raise ValueError("Model being loaded was trained WITHOUT modality start/end tokens (mm_use_vid_start_end=False), but at inference the model will use nodality start/end tokens (mm_use_vid_start_end=True). Not expected, set use_token_modality_prefix=False in this script.")
        else:
            assert mm_use_vid_start_end == False, "If mm_use_vid_start_end is True, then model.embed_tokens.weight should be in projector_weights"
        
        # Load the projection weights
        status = model.load_state_dict(projector_weights, strict=False)
        
        if status.unexpected_keys:
            print(f"Unexpected Keys: {status.unexpected_keys}.")
        print(f"Weights loaded from {projection_path}")
    else:
        raise ValueError("Projection path is required for loading weights. Comment this out if you want to evaluate base LLaVA.")

    # '''
    # Start Manish's code from csgpu7
    # '''
    # tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
    # if mm_use_vid_start_end:
    #     tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
    #     additional_tokens = [DEFAULT_OBJECT_START_TOKEN,DEFAULT_OBJECT_PATCH_TOKEN,DEFAULT_OBJECT_END_TOKEN,DEFAULT_POSE_START_TOKEN,DEFAULT_POSE_END_TOKEN,DEFAULT_POSE_PATCH_TOKEN]
    #     tokenizer.add_tokens(additional_tokens, special_tokens=True)
    # # Resize token embeddings of the model
    # model.resize_token_embeddings(len(tokenizer))

    # # Load the weights from projection_path after resizing the token_embeddings
    # if projection_path:
    #     print(f"Loading weights from {projection_path}")
    #     status = model.load_state_dict(torch.load(projection_path, map_location='cpu'), strict=False)
    #     if status.unexpected_keys:
    #         print(f"Unexpected Keys: {status.unexpected_keys}.\nThe skilvlm weights are not loaded correctly.")
    #     print(f"Weights loaded from {projection_path}")
    # '''
    # End Manish's code from csgpu7
    # '''

    # Set model to evaluation mode and move to GPU
    model = model.eval()
    model = model.cuda()
    #breakpoint()
    vision_tower_name = "openai/clip-vit-large-patch14"

    # Load vision tower and move to GPU
    vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name, torch_dtype=torch.float16,
                                                   low_cpu_mem_usage=True).cuda()
    vision_tower = vision_tower.eval()

    # Configure vision model
    vision_config = model.get_model().vision_config
    vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]   
    
    vision_config.use_vid_start_end = mm_use_vid_start_end
    if mm_use_vid_start_end:
        vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN])
        
    # string modality prefix. will be taken care of in the model inference code
    vision_config.use_string_modality_prefix = use_string_modality_prefix

    # Set video token length
    video_token_len = 356

    # print(tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN, DEFAULT_OBJECT_PATCH_TOKEN, DEFAULT_POSE_PATCH_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, DEFAULT_OBJECT_START_TOKEN, DEFAULT_OBJECT_END_TOKEN, DEFAULT_POSE_START_TOKEN, DEFAULT_POSE_END_TOKEN]))

    return model, vision_tower, tokenizer, image_processor, video_token_len
