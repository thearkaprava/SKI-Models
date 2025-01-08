import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence
import torch
import transformers
from torch.utils.data import Dataset
from skilvlm.train.llava_trainer import SKILVLMTrainer
from skilvlm import video_conversation as conversation_lib
from skilvlm.model import *
import torch.distributed as dist
from skilvlm.constants import *
import pickle
import numpy as np
import os

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    # use_modality_token_prefix: bool = field(default=False)
    use_modality_string_prefix: bool = field(default=False)
    mm_use_vid_start_end: bool = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    sep_video_conv_front: bool = False
    video_token_len: int = 0
    video_folder: Optional[str] = field(default=None)
    frame_aspect_ratio: str = 'square'

@dataclass
class SKILVLMArguments:
    object_folder: Optional[str] = field(default=None)
    pose_folder: Optional[str] = field(default=None)
    video_mask_prob: float = field(default=0.0)
    # object_mask_prob: float = field(default=0.0)
    pose_mask_prob: float = field(default=0.0)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    dist.barrier()
    state_dict = trainer.model.state_dict()
    trainer._save(output_dir, state_dict=state_dict)  # noqa

def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

# where modality placeholders are replaced with the start/patch/end tokens
def preprocess_multimodal(
        sources: Sequence[str],
        multimodal_cfg: dict,
        cur_video_token_len: int,
        cur_pose_token_len: int, 
        # cur_object_token_len: int, 
) -> Dict:
    is_multimodal = multimodal_cfg['is_multimodal']

    if not is_multimodal:
        return sources
    
    video_token_len = cur_video_token_len
    pose_token_len = cur_pose_token_len
    # object_token_len = cur_object_token_len
    
    for source in sources:
        if multimodal_cfg['sep_video_conv_front']: 
            raise NotImplementedError("Not implemented")
            assert DEFAULT_VIDEO_TOKEN in source[0]['value']
            source[0]['value'] = source[0]['value'].replace(DEFAULT_VIDEO_TOKEN, '').strip()
            source[0]['value'] = DEFAULT_VIDEO_TOKEN + conversation_lib.default_conversation.sep + \
                                 conversation_lib.default_conversation.roles[0] + ": " + source[0]['value']
        
        ## Adding video token patches
        for sentence in source:
            if cur_video_token_len > 0:
                replace_token = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len

                if multimodal_cfg['use_vid_start_end']:
                    replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN
                if multimodal_cfg['use_modality_string_prefix']:
                    replace_token = DEFAULT_VIDEO_STRING_PREFIX + replace_token

                sentence["value"] = sentence["value"].replace(DEFAULT_VIDEO_TOKEN, replace_token)
            else:
                sentence["value"] = sentence["value"].replace(DEFAULT_VIDEO_TOKEN, '')

        # ## Adding object token patches
        # for sentence in source:
        #     if cur_object_token_len > 0:
        #         num_objects = object_token_len // 8 # 8 frames per object
        #         replace_token = DEFAULT_OBJECT_PATCH_TOKEN * object_token_len

        #         if multimodal_cfg['use_vid_start_end']: # are we using modality prefix?
        #             replace_token = DEFAULT_OBJECT_START_TOKEN + replace_token + DEFAULT_OBJECT_END_TOKEN
        #         if multimodal_cfg['use_modality_string_prefix']:
        #             replace_token = DEFAULT_OBJECT_STRING_PREFIX + replace_token

        #         sentence["value"] = sentence["value"].replace(DEFAULT_OBJECT_TOKEN, replace_token)
        #     else:
        #         sentence["value"] = sentence["value"].replace(DEFAULT_OBJECT_TOKEN, '')

        ## Adding pose token patches
        for sentence in source:
            if cur_pose_token_len > 0:
                replace_token = DEFAULT_POSE_PATCH_TOKEN * pose_token_len

                if multimodal_cfg['use_vid_start_end']: 
                    replace_token = DEFAULT_POSE_START_TOKEN + replace_token + DEFAULT_POSE_END_TOKEN
                if multimodal_cfg['use_modality_string_prefix']:
                    replace_token = DEFAULT_POSE_STRING_PREFIX + replace_token

                sentence["value"] = sentence["value"].replace(DEFAULT_POSE_TOKEN, replace_token)
            else:
                sentence["value"] = sentence["value"].replace(DEFAULT_POSE_TOKEN, '')

    return sources

def preprocess_v1(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):

        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_mpt(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids) + len(tokenizer(conv.sep).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids)
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "v1":
        return preprocess_v1(sources, tokenizer)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)

    assert False, "Unexpected" # conversation_lib.default_conversation.version is v1 when I checked (Dominick)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 multimodal_cfg: dict):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.multimodal_cfg = multimodal_cfg

    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
    
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        cur_video_token_len = cur_pose_token_len = cur_object_token_len = 0

        # Load video features
        video_folder = self.multimodal_cfg['video_folder']
        if video_folder is not None and 'video' in sources[0]:
            video_file = self.list_data_dict[i]['video']
            with open(f"{video_folder}/{video_file}", "rb") as f:
                features = pickle.load(f)

            cur_video_token_len = 356  # 100 temporal + 256 spatial, TODO: Hard Coding is not good

            mask_prob = self.multimodal_cfg['modality_mask_probs']['video_mask_prob']
            if torch.rand(1) < mask_prob:
                features = features * 0
        
        # # Load object features
        # object_folder = self.multimodal_cfg['object_folder']
        # if object_folder is not None and 'object' in sources[0]: # Assuming object data is present
        #     object_file = self.list_data_dict[i]['object']
        #     if os.path.exists(f"{object_folder}/{object_file}"):
        #         with open(f"{object_folder}/{object_file}", "rb") as f:
        #             object_features = pickle.load(f)
        #             object_features = object_features.reshape(-1, 512)  # Reshape to [num_objects*8, 512] if it's not already
        #             object_features = torch.tensor(object_features, dtype=torch.float32)
        #     else:
        #         object_features = torch.zeros((8, 512), dtype=torch.float32)  # Create a zero tensor of shape [8, 512]

        #     cur_object_token_len = object_features.shape[0]

        #     mask_prob = self.multimodal_cfg['modality_mask_probs']['object_mask_prob']
        #     if torch.rand(1) < mask_prob:
        #         object_features = object_features * 0
    
        # Load pose features
        pose_folder = self.multimodal_cfg['pose_folder']
        if pose_folder is not None and 'pose' in sources[0]:
            pose_file = self.list_data_dict[i]['pose']
            if os.path.exists(f"{pose_folder}/{pose_file}"):
                with open(f"{pose_folder}/{pose_file}", "rb") as f:
                    pose_features = pickle.load(f)
                    pose_features = torch.tensor(pose_features, dtype=torch.float32)      
        
            else:
                # logging.error(f"Pose file not found: {pose_folder}/{pose_file}")
                pose_features = torch.zeros((75, 216), dtype=torch.float32)  # Placeholder tensor

            cur_pose_token_len= 75

            mask_prob = self.multimodal_cfg['modality_mask_probs']['pose_mask_prob']
            if torch.rand(1) < mask_prob:
                pose_features = pose_features * 0

        sources = preprocess_multimodal(
            copy.deepcopy([e["conversations"] for e in sources]),
            self.multimodal_cfg, cur_video_token_len, cur_pose_token_len) #, cur_object_token_len)
        
        data_dict = preprocess(
            sources,
            self.tokenizer)
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])
            
        if video_folder is not None and 'video' in self.list_data_dict[i]:
            data_dict["video"] = features
        
        # if object_folder is not None and 'object' in self.list_data_dict[i]:
        #     data_dict["object"] = object_features

        if pose_folder is not None and 'pose' in self.list_data_dict[i]:
            data_dict["pose"] = pose_features
    
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'video' in instances[0]: # TODO: Assuming pose is only present if video is present
            features = [torch.tensor(instance['video']) for instance in instances]

            if all(x is not None and x.shape == features[0].shape for x in features):
                batch['video_spatio_temporal_features'] = torch.stack(features)
            else:
                batch['video_spatio_temporal_features'] = features

        # if 'object' in instances[0]:
        #     object_features = [instance['object'] for instance in instances if 'object' in instance]
            
        #     # Pad the object features to the same length when videos have varying numbers of objects being tracked
        #     # NOTE: the padded features should not be used as input to the model, they should be spliced off in skilvlm_pose_object
        #     # before being passed to the model. This is only so we can batch the data.
        #     if object_features:
        #         batch['object_features'] = torch.nn.utils.rnn.pad_sequence(object_features, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        #     else:
        #         # If no valid object features are present, you might add a placeholder or handle it differently
        #         logging.info("No valid object features found among the batch instances.")

        if 'pose' in instances[0]:
            pose_features = [instance['pose'] for instance in instances]
            if all(x is not None and x.shape == pose_features[0].shape for x in pose_features):
                batch['pose_features'] = torch.stack(pose_features)
            else:
                batch['pose_features'] = pose_features

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                skilvlm_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (LazySupervisedDataset
                   if data_args.lazy_preprocess else SupervisedDataset)
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                multimodal_cfg=dict(
                                    is_multimodal=data_args.is_multimodal,
                                    sep_video_conv_front=data_args.sep_video_conv_front,
                                    video_token_len=data_args.video_token_len,
                                    video_folder=data_args.video_folder,
                                    # object_folder=skilvlm_args.object_folder,
                                    pose_folder=skilvlm_args.pose_folder,  # Pass the pose folder here
                                    frame_aspect_ratio=data_args.frame_aspect_ratio,
                                    use_vid_start_end=getattr(data_args, 'mm_use_vid_start_end', False),
                                    use_modality_string_prefix=getattr(data_args, 'use_modality_string_prefix', False),
                                    modality_mask_probs=dict(video_mask_prob=skilvlm_args.video_mask_prob,
                                                            #  object_mask_prob=skilvlm_args.object_mask_prob,
                                                             pose_mask_prob=skilvlm_args.pose_mask_prob)
                                    ))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, SKILVLMArguments))
    model_args, data_args, training_args, skilvlm_args = parser.parse_args_into_dataclasses()

    # multiline string
    print(f"""
    Modality Info:
    - Using Video: {data_args.video_folder is not None}
    # - Using Object: {skilvlm_args.object_folder is not None}
    - Using Pose: {skilvlm_args.pose_folder is not None}\n\n
    """)

    modality_info = {
        'video': True if data_args.video_folder is not None else False,
        # 'object': True if skilvlm_args.object_folder is not None else False,
        'pose': True if skilvlm_args.pose_folder is not None else False,
    }

    model = SKILVLMLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_args=modality_info,
    )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    model_vision_dict = model.get_model().initialize_vision_modules(
        modalities_to_use=modality_info,
        pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter
    )

    vision_config = model_vision_dict['vision_config']

    data_args.video_token_len = model_vision_dict['video_token_len']
    data_args.is_multimodal = True

    projector_names = ['mm_projector', 'mm_projector_forobject', 'mm_projector_forpose']
    # projectors = [model.get_model().mm_projector, model.get_model().mm_projector_forobject, model.get_model().mm_projector_forpose]
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)

        for projector_name in projector_names:
            if hasattr(model.get_model(), projector_name):
                projector = getattr(model.get_model(), projector_name)

                for p in projector.parameters():
                    p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for projector_name in projector_names:
            if hasattr(model.get_model(), projector_name):
                projector = getattr(model.get_model(), projector_name)

                for p in projector.parameters():
                    p.requires_grad = False

    model.config.mm_use_vid_start_end = data_args.mm_use_vid_start_end = model_args.mm_use_vid_start_end
    model.config.use_modality_string_prefix = data_args.use_modality_string_prefix = model_args.use_modality_string_prefix
    vision_config.use_vid_start_end = training_args.use_vid_start_end = model_args.mm_use_vid_start_end
    model.config.sep_video_conv_front = data_args.sep_video_conv_front

    model.initialize_vision_tokenizer(mm_use_vid_start_end=model_args.mm_use_vid_start_end, tokenizer=tokenizer,
                                      device=training_args.device, tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
                                      pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter)

    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'.format(
                    len(params_no_grad), params_no_grad))
            else:
                print(
                    '[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'.format(
                        len(params_no_grad), ', '.join(params_no_grad[:10])))
            print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
            print(
                "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)

                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, skilvlm_args=skilvlm_args) # this is the dataset
    training_args.report_to = []
    # training_args.max_steps = 10

    # show trainable parameters
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    logging.warning(f"!!! Make sure these are correct !!!\nTrainable Parameters: {trainable_params}")

    trainer = SKILVLMTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()