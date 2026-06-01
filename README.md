# Stable Diffusion Fine-Tuning with LoRA - Baseline vs Fine-Tuned Comparison
> ** Applied Machine Learning - Group 10 **
> Authors: Julia Włodarska, Sophia Sara Lopotaru, Teodora Dobreva

---

## Overview

This project investigates how LoRA (Low-Rank Adaptation) fine-tuning affects image generation quality in
Stable Diffusion, using the OpenFace-CQUPT dataset with human captions. We trained nine fine-tuned variants by
performing a grid search over three learning rates and three epoch counts. We compared the best of those models with the pre-trained basline using quantitive evaluation metrics.

---

## Dataset
The dataset consists of around 10 million samples of images found on the internet, with human captions explaining them (OpenFaceCQUPT). We used 12,000 randomly chosen samples from the dataset with human captions. The format of the images is JPEG.

Link to the original dataset: https://huggingface.co/datasets/OpenFace-CQUPT/FaceCaption-15M

Both images and the captions were preprocessed by removing images containing multiple individuals or depicting nudity, and excluding extremely low CLIP-scored images. We applied face detection to localize facial regions and used the resulting bounding boxes to crop each image, ensuring that only the face region was retained for subsequent processing. All the detailed preprocessing scripts are available in 'src/preprocessing/'. 

Link to the preprocessed dataset: https://huggingface.co/datasets/AML-group10/AML_project_preprocessed_dataset 

The examples of image-caption pairs included in the dataset after preprocessing are depicted below.

![Example preprocessed images](src/preprocessing/example_preprocessed.png)

---

## Model details

As the baseline, we used Segmind Tiny-SD, which is a lightweight distilled version of Stable Diffusion designed for fast and efficient image generation. Tiny-SD is derived from larger Stable Diffusion models through knowledge distillation. The model is trained on preprocessed text–image datasets and optimized for efficient deployment, making it suitable as a strong baseline for further fine-tuning and research in lightweight generative models.

Link to the Segmind Tiny Stable Diffusion baseline model: https://huggingface.co/segmind/tiny-sd

The code for training the model are included in 'src/models/training/'.

---

## Hyperparameter Tuning

The fine-tuninig method used was Low-Rank Adaptation (LoRA). UNet attentino layers were fine-tuned. Nine LoRA-adapted variants are stored under 'src/models/finetuned_models/', one per hyperparameter combination. Each fine-tuned model contains its LoRA weights and training config.

We performed a full grid search across following values:
| Hyperparameter | Values |
|---|---|
| Learning rate | `1e-4`, `5e-5`, `1e-5` |
| Number of epochs | `5`, `10`, `20` |
| **Total models** | **9** |

---

## Repository structure
AML-PROJECT/
├── archive/                 # Archived experiments
│   ├── cvae/                # CVAE-based experiments
│   ├── diffusion_models/    # Diffusion model experiments
│   ├── dog/                 # Dataset experiments
│   ├── lora-output/         # Test LoRA training outputs
│   └── tests/               # Experimental test scripts
│
├── deployment/             # Deployment utilities
│
├── src/
│   ├── evaluation/         # Evaluation and metric computation
│   ├── models/
│   │   ├── finetuned_models/  # 9 fine-tuned LoRA weights + configs
│   │   └── training/          # LoRA training scripts
│   │
│   ├── preprocessing/      # Data preprocessing pipelines
│   └── results/            # Generated outputs + experiment logs
│
├── README.md
├── requirements.txt
├── run_inference.sh
└── proposal.pdf

--- 

## Setup

1. Clone the repository:
```bash
    git clone https://github.com/AML-group10/AML-project.git
    cd AML-project
```

2. Install dependencies using uv:
```bash
    uv sync
```

---

## How to Run
### Training
Train a single fine-tuned model by specifying a learning rate and number of epochs:

```bash
    python src/train/lora_train.py \
    --base_model models/baseline/ \
    --data_dir data/processed/ \
    --learning_rate 1e-4 \
    --epochs 10 \
    --output_dir models/finetuned/lr1e-4_ep10/
```

### Inference
Generate single image from any model:

```bash
    python src/models/training/inference.py \
        --prompt "a man with curly black hair and blue eyes" \
        --output image.jpeg
```

---

## Results
Results are stored in 'src/results/'. A summary of validation metrics is as follows:

| Model | Learning Rate | Epochs | FID ↓ | CLIP Score ↑ |
|---|---|---|---|---|
| Baseline | — | — | — | — |
| Fine-tuned | 1e-4 | 5 | — | — |
| Fine-tuned | 1e-4 | 10 | — | — |
| Fine-tuned | 1e-4 | 20 | — | — |
| Fine-tuned | 5e-5 | 5 | — | — |
| Fine-tuned | 5e-5 | 10 | — | — |
| Fine-tuned | 5e-5 | 20 | — | — |
| Fine-tuned | 1e-5 | 5 | — | — |
| Fine-tuned | 1e-5 | 10 | — | — |
| Fine-tuned | 1e-5 | 20 | — | — |

FID (Frechet Inception Distance) measures image quality and diversity (the lower the better). CLIP Score measures prompt-image alignment (the higher the better).