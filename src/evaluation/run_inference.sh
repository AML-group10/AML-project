#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --job-name=inference_loop

python3 src/evaluation/inference_loop.py
