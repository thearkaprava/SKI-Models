from logging import Logger
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch
import numpy as np
from functools import partial
import random

import io
import os
import os.path as osp
import shutil
import warnings
from collections.abc import Mapping, Sequence
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import Dataset
import copy
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
import os.path as osp
import mmcv
import numpy as np
import torch
import tarfile
from .pipeline_AS import *
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from mmcv.parallel import collate
import pandas as pd

PIPELINES = Registry('pipeline')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self,
                 ann_file,
                 pipeline,
                 repeat = 1,
                 data_prefix=None,
                 test_mode=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0,
                 dynamic_length=False,
                 async_loading=False):
        super().__init__()
        self.use_tar_format = True if ".tar" in data_prefix else False
        data_prefix = data_prefix.replace(".tar", "")
        self.ann_file = ann_file
        self.repeat = repeat
        self.data_prefix = osp.realpath(
            data_prefix) if data_prefix is not None and osp.isdir(
                data_prefix) else data_prefix
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        self.sample_by_class = sample_by_class
        self.power = power
        self.dynamic_length = dynamic_length
        self.async_loading = async_loading

        assert not (self.multi_class and self.sample_by_class)

        self.pipeline = Compose(pipeline)

        ## Loading Hyperformer Embeddings'
        if 'ntu' in self.ann_file:
            # data: N C V T M
            data_path = '/data/ntu/Hyperformer_processed_data/NTU60_CS.npz'  #NTU60_CVS1.npz' NTU60_CS.npz  
            # data_path = '/data/ntu/Hyperformer_processed_data/NTU120_NTU120_110-10.npz' # NTU120_NTU120_110-10.npz NTU120_NTU120_96-24.npz
            npz_data = np.load(data_path)
            hyperformer_data_train = npz_data['x_train']
            hyperformer_label_train = np.where(npz_data['y_train'] > 0)[1]
            # self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
            hyperformer_filename_train = [x.decode('utf-8') for x in npz_data['files_train']]

            hyperformer_data_test = npz_data['x_test']
            hyperformer_label_test = np.where(npz_data['y_test'] > 0)[1]
            # self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
            hyperformer_filename_test = [x.decode('utf-8') for x in npz_data['files_test']]

            hyperformer_data = np.concatenate((hyperformer_data_train, hyperformer_data_test), axis = 0)
            hyperformer_label = np.concatenate((hyperformer_label_train, hyperformer_label_test), axis = 0)
            hyperformer_filename = np.concatenate((hyperformer_filename_train, hyperformer_filename_test), axis=0)

            N, T, _ = hyperformer_data.shape
            hyperformer_data = hyperformer_data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

            ## Dictionary for Hyperformer (keys:filenames; values:npz x_data)
            self.hyperformer_data = dict(zip(hyperformer_filename, hyperformer_data))

        self.video_infos = self.load_annotations()
        if self.sample_by_class:
            self.video_infos_by_class = self.parse_by_class()

            class_prob = []
            for _, samples in self.video_infos_by_class.items():
                class_prob.append(len(samples) / len(self.video_infos))
            class_prob = [x**self.power for x in class_prob]

            summ = sum(class_prob)
            class_prob = [x / summ for x in class_prob]

            self.class_prob = dict(zip(self.video_infos_by_class, class_prob))

    @abstractmethod
    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""

    # json annotations already looks like video_infos, so for each dataset,
    # this func should be the same
    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            if self.data_prefix is not None:
                path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value
            if self.multi_class:
                assert self.num_classes is not None
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]
        return video_infos

    def parse_by_class(self):
        video_infos_by_class = defaultdict(list)
        for item in self.video_infos:
            label = item['label']
            video_infos_by_class[label].append(item)
        return video_infos_by_class

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return mmcv.dump(results, out)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['filename'] = self.video_infos[idx]['filename']

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        aug1 = self.pipeline(results)
        aug1['filename'] = self.video_infos[idx]['filename']
        if self.repeat > 1:
            aug2 = self.pipeline(results)
            ret = {"imgs": torch.cat((aug1['imgs'], aug2['imgs']), 0),
                    "label": aug1['label'].repeat(2),
            }
            return ret
        else:
            return aug1

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        
        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot
        
        results = self.pipeline(results)
        results['filename'] = self.video_infos[idx]['filename']
        
        return results

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_frames(idx)

        return self.prepare_train_frames(idx)

class VideoDataset(BaseDataset):
    def __init__(self, ann_file, pipeline, labels_file, start_index=0, **kwargs):
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)
        self.labels_file = labels_file
        if 'async_loading' in list(kwargs.keys()):
            self.async_loading = kwargs['async_loading']
        else:    
            self.async_loading = False

    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()

        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                # check if the line is comma separated instead of space separated
                if len(line_split) == 1 and ',' in line_split[0]:
                    line_split = line_split[0].split(',')
                
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                elif len(line_split) == 3: # assume the format is rgb_path, pose_path, label
                    filename, pose_path, label = line_split
                    label = int(label)
                else:
                    filename, label = line_split
                    label = int(label)
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)

                ### Get corresponding hyperformer embedding
                if 'ntu' in self.ann_file:
                    # '/data/ntu/NTU60_224x224/rgb/S014C002P027R002A019_rgb.avi'
                    _, fname = os.path.split(filename)
                    video_identifier = fname[:-8]
                else:
                    raise NotImplementedError('filename -> hyperformer dict index')

                if self.async_loading:
                    ############################################
                    ### Asynchronous loading of hyperformer data
                    # Get the last 4 characters of the video_identifier

                    # print("########## ASYNCHRONOUS TRAINING ###########")
                    action_class = video_identifier[-4:]

                    # Filter filenames that have the last 4 characters equal to action_class
                    filtered_filenames = [filename for filename in self.hyperformer_data.keys() if filename[-4:] == action_class]

                    # Remove the video_identifier from the filtered filenames
                    filtered_filenames.remove(video_identifier)

                    # Randomly choose a filename from the filtered filenames
                    random_filename_same_action = random.choice(filtered_filenames)
                    hformer_data = self.hyperformer_data[random_filename_same_action]

                else:
                    hformer_data = self.hyperformer_data[video_identifier]
            
                video_infos.append(dict(filename=filename, label=label, hformer_data=hformer_data, tar=self.use_tar_format))
        return video_infos


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


# def mmcv_collate(batch, samples_per_gpu=1):
#     print(f"Batch Type: {type(batch)}")
#     if isinstance(batch, Sequence):
#         if isinstance(batch[0], Sequence):
#             print(f"First item type in batch: {type(batch[0])}")
#         elif isinstance(batch[0], Mapping):
#             print(f"First item type in batch: {type(batch[0])}")
#         else:
#             print(f"Item in batch: {batch[0]}")
#     else:
#         print(f"Unexpected batch type: {type(batch)}")
    
#     # Proceed with collate as usual
#     if not isinstance(batch, Sequence):
#         raise TypeError(f'{type(batch)} is not supported.')

#     if isinstance(batch[0], Sequence) and not isinstance(batch[0], Mapping):
#         transposed = zip(*batch)
#         return [mmcv_collate(samples, samples_per_gpu) for samples in transposed]
#     elif isinstance(batch[0], Mapping):
#         collated_batch = {}
#         for key in batch[0]:
#             collated_batch[key] = mmcv_collate([d[key] for d in batch], samples_per_gpu)
#         return collated_batch
#     else:
#         return default_collate(batch)


## original
# def mmcv_collate(batch, samples_per_gpu=1): 
#     if not isinstance(batch, Sequence):
#         raise TypeError(f'{batch.dtype} is not supported.')
#     if isinstance(batch[0], Sequence):
#         transposed = zip(*batch)
#         return [collate(samples, samples_per_gpu) for samples in transposed]
#     elif isinstance(batch[0], Mapping):
#         return {
#             key: mmcv_collate([d[key] for d in batch], samples_per_gpu)
#             for key in batch[0]
#         }
#     else:
#         return default_collate(batch)

def mmcv_collate(batch, samples_per_gpu=1):
    if not isinstance(batch, Sequence):
        raise TypeError(f'{type(batch)} is not supported.')

    # If the first item is a sequence, handle as a list of lists
    if isinstance(batch[0], Sequence) and not isinstance(batch[0], Mapping):
        transposed = zip(*batch)
        return [mmcv_collate(samples, samples_per_gpu) for samples in transposed]
    # If the first item is a dictionary, handle as a list of dictionaries
    elif isinstance(batch[0], Mapping):
        return {
            key: mmcv_collate([d[key] for d in batch], samples_per_gpu)
            if isinstance(batch[0][key], (Mapping, Sequence)) and not isinstance(batch[0][key], str)
            else default_collate([d[key] for d in batch])
            for key in batch[0]
        }
    # If the items are neither sequences nor dictionaries, handle normally
    else:
        return default_collate(batch)

# def mmcv_collate(batch, samples_per_gpu=1):
#     if not isinstance(batch, Sequence):
#         raise TypeError(f'{type(batch)} is not supported.')

#     batch_dict = {}
#     # Initialize containers for all keys
#     for key in batch[0]:
#         batch_dict[key] = []

#     # Aggregate values for each key from all samples in the batch
#     for data in batch:
#         for key, value in data.items():
#             batch_dict[key].append(value)

#     # Collate each item in the dictionary appropriately
#     for key in batch_dict:
#         if key == 'filename' or key == 'label':  # Handle filenames and labels as lists
#             continue  # Already appropriately collected, no further processing needed
#         elif isinstance(batch_dict[key][0], torch.Tensor):
#             batch_dict[key] = torch.stack(batch_dict[key], dim=0)  # Stack tensors for tensor type data
#         else:
#             # Use default collate for all other types
#             batch_dict[key] = torch.utils.data.dataloader.default_collate(batch_dict[key])

#     return batch_dict



def build_dataloader(logger, config):
    scale_resize = int(256 / 224 * config.DATA.INPUT_SIZE)

    train_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(
            type='MultiScaleCrop',
            input_size=config.DATA.INPUT_SIZE,
            scales=(1, 0.875, 0.75, 0.66),
            random_crop=False,
            max_wh_scale_gap=1),
        dict(type='Resize', scale=(config.DATA.INPUT_SIZE, config.DATA.INPUT_SIZE), keep_ratio=False),
        dict(type='Flip', flip_ratio=0.5),
        dict(type='ColorJitter', p=config.AUG.COLOR_JITTER),
        dict(type='GrayScale', p=config.AUG.GRAY_SCALE),
        # dict(type='PreprocessPose', split='train', p_interval=[], bone=True),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label', 'hformer_data'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label', 'hformer_data']),        
        # dict(type='Collect', keys=['imgs', 'label', 'hformer_embeds'], meta_keys=[]),
        # dict(type='ToTensor', keys=['imgs', 'label', 'hformer_embeds']),
    ]
    train_data = VideoDataset(ann_file=config.DATA.TRAIN_FILE, data_prefix=config.DATA.ROOT,
                              labels_file=config.DATA.LABEL_LIST, pipeline=train_pipeline, async_loading=config.DATA.ASYNC_LOADING)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    train_loader = DataLoader(
        train_data, sampler=sampler_train,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(mmcv_collate, samples_per_gpu=config.TRAIN.BATCH_SIZE),
        # collate_fn=default_collate,
    )
    
    val_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, test_mode=True),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
        # dict(type='PreprocessPose', split='test', p_interval=[], bone=True),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),        
        dict(type='Collect', keys=['imgs', 'label', 'hformer_data'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label', 'hformer_data']),  
        # dict(type='Collect', keys=['imgs', 'label', 'hformer_embeds'], meta_keys=[]),
        # dict(type='ToTensor', keys=['imgs', 'hformer_embeds'])
    ]
    if config.TEST.NUM_CROP == 3:
        val_pipeline[3] = dict(type='Resize', scale=(-1, config.DATA.INPUT_SIZE))
        val_pipeline[4] = dict(type='ThreeCrop', crop_size=config.DATA.INPUT_SIZE)
    if config.TEST.NUM_CLIP > 1:
        val_pipeline[1] = dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, multiview=config.TEST.NUM_CLIP)
    
    val_data = VideoDataset(ann_file=config.DATA.VAL_FILE, data_prefix=config.DATA.ROOT, labels_file=config.DATA.LABEL_LIST, pipeline=val_pipeline)
    indices = np.arange(dist.get_rank(), len(val_data), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)
    val_loader = DataLoader(
        val_data, sampler=sampler_val,
        batch_size=2,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(mmcv_collate, samples_per_gpu=2),
        # collate_fn=partial(default_collate, samples_per_gpu=2),
        # collate_fn=default_collate,
    )

    return train_data, val_data, train_loader, val_loader 

## Dataloader without DDP for late fusion with preprocessPose
def build_dataloader_no_ddp(logger, config):
    scale_resize = int(256 / 224 * config.DATA.INPUT_SIZE)

    train_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(
            type='MultiScaleCrop',
            input_size=config.DATA.INPUT_SIZE,
            scales=(1, 0.875, 0.75, 0.66),
            random_crop=False,
            max_wh_scale_gap=1),
        dict(type='Resize', scale=(config.DATA.INPUT_SIZE, config.DATA.INPUT_SIZE), keep_ratio=False),
        dict(type='Flip', flip_ratio=0.5),
        dict(type='ColorJitter', p=config.AUG.COLOR_JITTER),
        dict(type='GrayScale', p=config.AUG.GRAY_SCALE),
        # dict(type='PreprocessPose', split='train', p_interval=[], bone=False),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label', 'hformer_data'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label', 'hformer_data']),        
        # dict(type='Collect', keys=['imgs', 'label', 'hformer_embeds'], meta_keys=[]),
        # dict(type='ToTensor', keys=['imgs', 'label', 'hformer_embeds']),
    ]
        
    
    train_data = VideoDataset(ann_file=config.DATA.TRAIN_FILE, data_prefix=config.DATA.ROOT,
                              labels_file=config.DATA.LABEL_LIST, pipeline=train_pipeline)
    # num_tasks = dist.get_world_size()
    # global_rank = dist.get_rank()
    # sampler_train = torch.utils.data.DistributedSampler(
    #     train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
    # )
    train_loader = DataLoader(
        train_data, 
        # sampler=sampler_train,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        # collate_fn=partial(mmcv_collate, samples_per_gpu=config.TRAIN.BATCH_SIZE),
    )
    
    val_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, test_mode=True),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
        # dict(type='PreprocessPose', split='test', p_interval=[], bone=False),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),        
        dict(type='Collect', keys=['imgs', 'label', 'hformer_data'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label', 'hformer_data']),  
        # dict(type='Collect', keys=['imgs', 'label', 'hformer_embeds'], meta_keys=[]),
        # dict(type='ToTensor', keys=['imgs', 'hformer_embeds'])
    ]
    if config.TEST.NUM_CROP == 3:
        val_pipeline[3] = dict(type='Resize', scale=(-1, config.DATA.INPUT_SIZE))
        val_pipeline[4] = dict(type='ThreeCrop', crop_size=config.DATA.INPUT_SIZE)
    if config.TEST.NUM_CLIP > 1:
        val_pipeline[1] = dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, multiview=config.TEST.NUM_CLIP)
    
    val_data = VideoDataset(ann_file=config.DATA.VAL_FILE, data_prefix=config.DATA.ROOT, labels_file=config.DATA.LABEL_LIST, pipeline=val_pipeline)
    # indices = np.arange(dist.get_rank(), len(val_data), dist.get_world_size())
    # sampler_val = SubsetRandomSampler(indices)
    val_loader = DataLoader(
        val_data, 
        # sampler=sampler_val,
        batch_size=2,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        # collate_fn=partial(mmcv_collate, samples_per_gpu=2),
    )

    return train_data, val_data, train_loader, val_loader 

def build_dataloader_no_ddp_bone(logger, config):
    print("Bone")
    scale_resize = int(256 / 224 * config.DATA.INPUT_SIZE)

    train_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(
            type='MultiScaleCrop',
            input_size=config.DATA.INPUT_SIZE,
            scales=(1, 0.875, 0.75, 0.66),
            random_crop=False,
            max_wh_scale_gap=1),
        dict(type='Resize', scale=(config.DATA.INPUT_SIZE, config.DATA.INPUT_SIZE), keep_ratio=False),
        dict(type='Flip', flip_ratio=0.5),
        dict(type='ColorJitter', p=config.AUG.COLOR_JITTER),
        dict(type='GrayScale', p=config.AUG.GRAY_SCALE),
        dict(type='PreprocessPose', split='train', p_interval=[], bone=True),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label', 'hformer_data'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label', 'hformer_data']),        
        # dict(type='Collect', keys=['imgs', 'label', 'hformer_embeds'], meta_keys=[]),
        # dict(type='ToTensor', keys=['imgs', 'label', 'hformer_embeds']),
    ]
        
    
    train_data = VideoDataset(ann_file=config.DATA.TRAIN_FILE, data_prefix=config.DATA.ROOT,
                              labels_file=config.DATA.LABEL_LIST, pipeline=train_pipeline)
    # num_tasks = dist.get_world_size()
    # global_rank = dist.get_rank()
    # sampler_train = torch.utils.data.DistributedSampler(
    #     train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
    # )
    train_loader = DataLoader(
        train_data, 
        # sampler=sampler_train,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        # collate_fn=partial(mmcv_collate, samples_per_gpu=config.TRAIN.BATCH_SIZE),
    )
    
    val_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, test_mode=True),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
        dict(type='PreprocessPose', split='test', p_interval=[], bone=True),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),        
        dict(type='Collect', keys=['imgs', 'label', 'hformer_data'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label', 'hformer_data']),  
        # dict(type='Collect', keys=['imgs', 'label', 'hformer_embeds'], meta_keys=[]),
        # dict(type='ToTensor', keys=['imgs', 'hformer_embeds'])
    ]
    if config.TEST.NUM_CROP == 3:
        val_pipeline[3] = dict(type='Resize', scale=(-1, config.DATA.INPUT_SIZE))
        val_pipeline[4] = dict(type='ThreeCrop', crop_size=config.DATA.INPUT_SIZE)
    if config.TEST.NUM_CLIP > 1:
        val_pipeline[1] = dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, multiview=config.TEST.NUM_CLIP)
    
    val_data = VideoDataset(ann_file=config.DATA.VAL_FILE, data_prefix=config.DATA.ROOT, labels_file=config.DATA.LABEL_LIST, pipeline=val_pipeline)
    # indices = np.arange(dist.get_rank(), len(val_data), dist.get_world_size())
    # sampler_val = SubsetRandomSampler(indices)
    val_loader = DataLoader(
        val_data, 
        # sampler=sampler_val,
        batch_size=2,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        # collate_fn=partial(mmcv_collate, samples_per_gpu=2),
    )

    return train_data, val_data, train_loader, val_loader 