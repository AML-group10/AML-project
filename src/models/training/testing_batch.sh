#!/bin/bash

python3 src/models/training/lora_training.py \
    --pretrained_model_name_or_path="segmind/tiny-sd" \
    --dataset_name="AML-group10/AML_project_preprocessed_dataset" \
    --dataset_config_name="train" \
    --output_dir="./AML-group10/testing_batch_size" \
    --use_peft \
    --lora_r=4 \
    --lora_alpha=4 \
    --resolution=512 \
    --train_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=100 \
    --learning_rate=1e-4 \
    --validation_prompt="people with toothbrushes" \
    --seed=67
