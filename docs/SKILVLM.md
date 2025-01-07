# SKI-LVLM: Skeleton-Induced Large Vision Language Models


This codebase is adapted from [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT).

## Installation
Our python environement is identical to [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT), we recommend following their installation instructions:

```shell
conda create --name=skilvlm_env python=3.10
conda activate skilvlm_env

git clone https://github.com/thearkaprava/SKI-Models.git
cd SKI_LVLM
pip install -r requirements.txt

export PYTHONPATH="./:$PYTHONPATH"
```

Additionally, if you are using A100/H100 GPUs you can install [FlashAttention](https://github.com/HazyResearch/flash-attention),
```shell
pip install ninja

git clone https://github.com/HazyResearch/flash-attention.git
cd flash-attention
git checkout v1.0.7
python setup.py install
```
---

## Training 

SKI-LVLM is trained on 100K video-instruction pairs generated from single-frame captions of each video in the NTU-RGB+D dataset. The weights of the model are initialized from LLaVA and it is trained for 3 epochs on 8 48GB NVIDIA RTX A6000 GPUs. To begin, download the LLaVA weights from this link: [LLaVA-7B-Lightening-v1-1](https://huggingface.co/mmaaz60/LLaVA-7B-Lightening-v1-1).

**Download the pre-extracted features**:
   - Download the Multi-modal Features (`video_features.zip`, `pose_features.zip`) and Instruction Dataset (`NTU_QA-for-training.json`)
   - This should result in separate directories for each modality, and a json for training

### Training 
The command below will train the SKI-LVLM architecture for 3 epochs on video and skeleton features. This command is modular and will only train SKI-LVLM with the modalities whose folders are passed. For example, if only `--video_folder` is passed, SKI-LVLM will drop the video modality and will only train with the video modality.
```shell
torchrun --nproc_per_node=8 --master_port 29001 llavidal/train/train_mem.py \
          --version v1 \
          --tune_mm_mlp_adapter True \
          --mm_use_vid_start_end \
          --bf16 True \
          --num_train_epochs 3 \
          --per_device_train_batch_size 4 \
          --per_device_eval_batch_size 4 \
          --gradient_accumulation_steps 1 \
          --evaluation_strategy "no" \
          --save_strategy "steps" \
          --save_steps 3000 \
          --save_total_limit 3 \
          --learning_rate 2e-5 \
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --lr_scheduler_type "cosine" \
          --logging_steps 100 \
          --tf32 True \
          --model_max_length 2048 \
          --gradient_checkpointing True \
          --lazy_preprocess True \
          --output_dir ./work_dirs/LLAVIDAL_video-object-pose-text_3epochs \
          --model_name_or_path /path/to/LLaVA-7B-Lightening-v-1-1/ \
          --data_path /path/to/NTU_QA-for-training.json \
          --video_folder /path/to/video_features/ \
          --pose_folder /path/to/pose_features/
```

## Quantitative Evaluation 🧪


Step 1: Download the dataset - [Charades](https://prior.allenai.org/projects/charades).

Step 2: Evaulate SKI-LVLM:

```shell
cd evaluation/ADL-Descriptions/eval
```
```shell
bash evaluate_all_benchmark_descriptions.sh
```
Pass the appropiate paths to get the results json.

---

