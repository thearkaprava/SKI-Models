## Stage 1 - video only
torchrun --nproc_per_node=8 --master_port 29001 llavidal/train/train_mem.py --model_name_or_path /path/to/LLaVA-7B-Lightening-v1-1 --version v1 --data_path /path/to/NTU_QA-for-training.json --tune_mm_mlp_adapter True --bf16 True --num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --evaluation_strategy "no" --save_strategy "steps" --save_steps 3000 --save_total_limit 3 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 100 --tf32 True --model_max_length 2048 --gradient_checkpointing True --lazy_preprocess True --video_folder /path/to/video_features/ --output_dir ./work_dirs/mmpose_training/stage1-video-text/ --mm_use_vid_start_end

## Stage 1 - object only
torchrun --nproc_per_node=8 --master_port 29001 llavidal/train/train_mem.py --model_name_or_path /path/to/LLaVA-7B-Lightening-v1-1 --version v1 --data_path /path/to/NTU_QA-for-training.json --tune_mm_mlp_adapter True --bf16 True --num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --evaluation_strategy "no" --save_strategy "steps" --save_steps 3000 --save_total_limit 3 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 100 --tf32 True --model_max_length 2048 --gradient_checkpointing True --lazy_preprocess True --object_folder /path/to/object_features/ --output_dir ./work_dirs/mmpose_training/stage1-object-text --mm_use_vid_start_end

## Stage 1 - pose only
torchrun --nproc_per_node=8 --master_port 29001 llavidal/train/train_mem.py --model_name_or_path /path/to/LLaVA-7B-Lightening-v1-1 --version v1 --data_path /path/to/NTU_QA-for-training.json --tune_mm_mlp_adapter True --bf16 True --num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --evaluation_strategy "no" --save_strategy "steps" --save_steps 3000 --save_total_limit 3 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 100 --tf32 True --model_max_length 2048 --gradient_checkpointing True --lazy_preprocess True --pose_folder /path/to/pose_features/ --output_dir ./work_dirs/mmpose_training/stage1-pose-text --mm_use_vid_start_end



# ## Merge stage 1 weights for video and pose
python merge_weights.py --weights ./work_dirs/mmpose_training/stage1-video-text/mm_projector.bin ./work_dirs/mmpose_training/stage1-pose-text/mm_projector.bin --output ./work_dirs/mmpose_training/stage2-initialization_video-pose-text.bin



## Start stage 2 training - video and pose
torchrun --nproc_per_node=8 --master_port 29001 llavidal/train/train_mem.py --model_name_or_path /path/to/LLaVA-7B-Lightening-v1-1 --version v1 --data_path /path/to/NTU_QA-for-training.json --tune_mm_mlp_adapter True --bf16 True --num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --evaluation_strategy "no" --save_strategy "steps" --save_steps 3000 --save_total_limit 3 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 100 --tf32 True --model_max_length 2048 --gradient_checkpointing True --lazy_preprocess True --video_folder /path/to/video_features/ --pose_folder /path/to/pose_features/ --output_dir ./work_dirs/mmpose_training/stage2_video-pose-text/ --mm_use_vid_start_end --pretrain_mm_mlp_adapter ./work_dirs/mmpose_training/stage2-initialization_video-pose-text.bin



# ## Merge stage 2 weights for video-pose and object
python merge_weights.py --weights ./work_dirs/mmpose_training/stage1-object-text/mm_projector.bin ./work_dirs/mmpose_training/stage2_video-pose-text/ --output ./work_dirs/mmpose_training/stage3_initialization_video-pose-object-text.bin



## Start stage 3 training - video, object and pose -- CHANGE pretrain_mm_mlp_adapter TO POINT TO CORRECT STAGE 2 WEIGHTS
torchrun --nproc_per_node=8 --master_port 29001 llavidal/train/train_mem.py --model_name_or_path /path/to/LLaVA-7B-Lightening-v1-1 --version v1 --data_path /path/to/NTU_QA-for-training.json --tune_mm_mlp_adapter True --bf16 True --num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --evaluation_strategy "no" --save_strategy "steps" --save_steps 3000 --save_total_limit 3 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 100 --tf32 True --model_max_length 2048 --gradient_checkpointing True --lazy_preprocess True --video_folder /path/to/video_features/ --object_folder /path/to/object_features/ --pose_folder /path/to/pose_features/ --output_dir ./work_dirs/mmpose_training/stage3_video-pose-object-text/ --mm_use_vid_start_end --pretrain_mm_mlp_adapter ./work_dirs/mmpose_training/stage3_initialization_video-pose-object-text.bin
