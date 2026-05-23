#!/bin/bash

learning_rates=(1e-4 3e-4 5e-4 1e-3)

for lr in "${learning_rates[@]}"; do
    echo "Learning rate $lr"

    python3 lora_training.py \
    --pretrained_model_name_or_path="segmind/tiny-sd" 
    --train_data_dir="./AML-group10/AML_project_preprocessed_dataset" 
    --output_dir="./AML-group10/${lr}" 
    --use_peft 
    --lora_r=4 
    --lora_alpha=4 
    --resolution=512 
    --train_batch_size=1 
    --gradient_accumulation_steps=4 
    --num_train_epochs=100 
    --learning_rate=$lr 
    --validation_prompt="people with toothbrushes" 
    --seed=67

done