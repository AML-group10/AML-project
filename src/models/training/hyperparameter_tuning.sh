#!/bin/bash

learning_rates=(1e-4 3e-4 5e-4)
num_epochs=(10 15 20)

for lr in "${learning_rates[@]}"; do
    for epochs in "${num_epochs[@]}"; do
        echo "Learning rate $lr, epochs $epochs"

        python3 src/models/training/lora_training.py \
        --pretrained_model_name_or_path="segmind/tiny-sd" \
        --dataset_name="AML-group10/AML_project_preprocessed_dataset" \
        --dataset_config_name="train" \
        --output_dir="./AML-group10/${lr}_${epochs}_hyperparameter_tuning" \
        --use_peft \
        --lora_r=4 \
        --lora_alpha=4 \
        --resolution=256 \
        --train_batch_size=512 \
        --gradient_accumulation_steps=1 \
        --num_train_epochs=$epochs \
        --learning_rate=$lr \
        --caption_column="prompt" \
        --push_to_hub \
        --allow_tf32 \
        --validation_epochs \
        --validation_prompt="a man with curly black hair, blue eyes and a moustache" \
        --seed=67
done