<!-- # Fine-tuned CLIP models are efficient video learners [CVPR 2023] -->
# <u>SKI</u> Models: <u>SK</u>eleton <u>I</u>nduced Vision-Language Embeddings for Understanding Activities of Daily Living

This is the official repository of 'SKI Models: SKeleton Induced Vision-Language Embeddings for Understanding Activities of Daily Living'.

## Installation
This codebase is tested on Ubuntu 20.04.2 LTS with python 3.8. Follow the below steps to create environment and install dependencies.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -y -n ski_env python=3.7
# Activate the environment
conda activate ski_env
# Install requirements
pip install -r requirements.txt
```

* Install Apex for enabling mixed-precision training.

NOTE: Make sure to have system CUDA of same version as of PyTorch CUDA version to properly install apex.
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


# Training
For all experiments shown in above tables, we provide config files in `configs` folder. For example, to train ViFi-CLIP (tunes both image and text encoder) on Kinetics-400, run the following command:
```
python -m torch.distributed.launch --nproc_per_node=8 \
main.py -cfg /PATH/TO/CONFIG --output /PATH/TO/OUTPUT
```

**Note:**
- We recommend keeping the total batch size as mentioned in respective config files. Please use `--accumulation-steps` to maintain the total batch size. Specifically, here the effective total batch size is 8(`GPUs_NUM`) x 4(`TRAIN.BATCH_SIZE`) x 16(`TRAIN.ACCUMULATION_STEPS`) = 512.
- After setting up the datasets as instructed [DATASETS.md](docs/DATASETS.md), only argument in the config file that should be specified is data path. All other settings in config files are pre-set.

For detailed training instructions for all experimental setup, please refer to [TRAIN.md](docs/TRAIN.md).

# Evaluating models
To evaluate a model, please use a suitable config and corresponding model weights. For example, to evaluate ViFi-CLIP with 16 frames on Kinetics-400, run the command below:
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
-cfg /PATH/TO/CONFIG --output /PATH/TO/OUTPUT \
--only_test --resume /PATH/TO/CKPT --opts TEST.NUM_CLIP 4 TEST.NUM_CROP 3