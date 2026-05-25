#!/bin/bash
lr=5e-4

python3 src/models/training/lora_training.py \
    --pretrained_model_name_or_path="segmind/tiny-sd" \
    --dataset_name="AML-group10/AML_project_preprocessed_dataset" \
    --dataset_config_name="train" \
    --output_dir="./AML-group10/${lr}_30hyperparameter_tuning" \
    --use_peft \
    --lora_r=4 \
    --lora_alpha=4 \
    --resolution=256 \
    --train_batch_size=512 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=30 \
    --learning_rate=$lr \
    --caption_column="prompt" \
    --push_to_hub \
    --validation_epochs=1 \
    --allow_tf32 \
    --validation_prompt="a man with curly black hair, blue eyes and a moustache" \
    --seed=67
