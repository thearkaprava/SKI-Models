import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.ROOT = ''
_C.DATA.TRAIN_FILE = ''
_C.DATA.VAL_FILE = ''
_C.DATA.DATASET = 'kinetics400'
_C.DATA.INPUT_SIZE = 224
_C.DATA.NUM_FRAMES = 8
_C.DATA.NUM_CLASSES = 400
_C.DATA.LABEL_LIST = 'labels/kinetics_400_labels.csv'
_C.DATA.ASYNC_LOADING = False
_C.DATA.SKELETON_DATA = ''

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.ARCH = 'ViT-B/32'
_C.MODEL.DROP_PATH_RATE = 0.
_C.MODEL.PRETRAINED = None
_C.MODEL.RESUME = None
_C.MODEL.FIX_TEXT = True

_C.MODEL.POSE_MODEL = 'trainers.Hyperformer.Hyperformer_Model'
_C.MODEL.GRAPH = 'graph.ntu_rgb_d.Graph'
_C.MODEL.LABELING_MODE = 'spatial'
_C.MODEL.JOINT_LABEL = [0, 4, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 1, 0, 1, 0, 1]

_C.MODEL.RESUME_POSE = None

# -----------------------------------------------------------------------------
# Custom trainer settings
# -----------------------------------------------------------------------------
_C.TRAINER = CN()
# Config for SKI-VLM
_C.TRAINER.SKI_VLM = CN()
_C.TRAINER.SKI_VLM.PROMPT_MODEL = False # second stage prompting?
_C.TRAINER.SKI_VLM.N_CTX_VISION = 0  # number of context vectors at the vision branch
_C.TRAINER.SKI_VLM.N_CTX_TEXT = 0  # number of context vectors at the language branch
_C.TRAINER.SKI_VLM.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
_C.TRAINER.SKI_VLM.PROMPT_DEPTH_VISION = 0  # max 12, min 0, for 0 it will act as shallow vision prompting (first layer)
_C.TRAINER.SKI_VLM.PROMPT_DEPTH_TEXT = 1  # max 12, min 0, for 0 it will act as shallow language prompting (first layer)
_C.TRAINER.SKI_VLM.USE = "both"  # fine-tuning complete VLM model by default
_C.TRAINER.SKI_VLM.ZS_EVAL = False  # make True only during test mode to evaluate zero-shot vanilla VLM performance
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 30
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WEIGHT_DECAY = 0.001
_C.TRAIN.LR = 8.e-6
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.ACCUMULATION_STEPS = 1
_C.TRAIN.LR_SCHEDULER = 'cosine'
_C.TRAIN.OPTIMIZER = 'adamw'
_C.TRAIN.OPT_LEVEL = 'O1'
_C.TRAIN.AUTO_RESUME = False
_C.TRAIN.USE_CHECKPOINT = False

_C.TRAIN.LOSS_SCALE = 0.1 #0.35
_C.TRAIN.CLIP_GRAD = 1.0

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
_C.AUG.LABEL_SMOOTH = 0.1
_C.AUG.COLOR_JITTER = 0.8
_C.AUG.GRAY_SCALE = 0.2
_C.AUG.MIXUP = 0.8
_C.AUG.CUTMIX = 1.0
_C.AUG.MIXUP_SWITCH_PROB = 0.5

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.NUM_CLIP = 1
_C.TEST.NUM_CROP = 1
_C.TEST.ONLY_TEST = False
_C.TEST.MULTI_VIEW_INFERENCE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.OUTPUT = ''
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 50
_C.SEED = 1024



def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.config)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    # merge from specific arguments
    if args.alpha:
        config.TRAIN.LOSS_SCALE = args.alpha
    if args.batch_size:
        config.TRAIN.BATCH_SIZE = args.batch_size
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.resume_pose:
        config.MODEL.RESUME_POSE = args.resume_pose
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.output:
        config.OUTPUT = args.output
    if args.only_test:
        config.TEST.ONLY_TEST = True
    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config