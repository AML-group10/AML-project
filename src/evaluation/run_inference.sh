#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --job-name=inference_loop

source /scratch/s5965780/AML-project/.venv/bin/activate
cd /scratch/s5965780/AML-project
python3 src/evaluation/inference_loop.py
