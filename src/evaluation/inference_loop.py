from datasets import load_dataset
import os
from evaluation_functions import run_evaluation
from diffusers import DiffusionPipeline 
import torch

import sys
sys.path.append("/scratch/s5965780/AML-project/src")
from models.training.inference import load_and_set_lora_ckpt

# Load validation prompts from HuggingFace
device = "cpu"
dataset = load_dataset("AML-group10/AML_project_preprocessed_dataset", "valid", split="train")
dataset = dataset.shuffle(seed=42).select(range(700))
prompts = [item["prompt"][0] for item in dataset]

# Attributes for evaluation
attributes = {
    "male": ["male", "man", "boy", "guy"],
    "female": ["female", "woman", "girl", "lady"],
    "young": ["young", "child", "teen", "baby"],
    "old": ["old", "elderly"],
    "white": ["white", "caucasian"],
    "black": ["black", "african american"],
    "asian": ["asian", "chinese", "japanese", "korean"],
}

"""("AML-group10/1e-4_20_hyperparameter_tuning", 200), julia
    ("AML-group10/1e-4_15_hyperparameter_tuning", 150), julia
    ("AML-group10/1e-4_10_hyperparameter_tuning", 100), julia
    ("AML-group10/5e-4_10_hyperparameter_tuning", 100), teddy
    ("AML-group10/5e-4_15_hyperparameter_tuning", 150),teddy
    ("AML-group10/5e-4_20_hyperparameter_tuning", 200), teddy
    ("AML-group10/3e-4_10_hyperparameter_tuning", 100), sophie
    ("AML-group10/3e-4_15_hyperparameter_tuning", 150), sophie
    ("AML-group10/3e-4_20_hyperparameter_tuning", 200) sophie """

# Loop over all 9 models
models = [
    ("AML-group10/1e-4_20_hyperparameter_tuning", 200)
]

os.makedirs("real_validation", exist_ok=True)
os.makedirs("validation_results", exist_ok=True)

for i, item in enumerate(dataset):
    item["image"].save(f"real_validation/image_{i}.jpeg")

for model_name, step_count in models:
    folder_name = model_name.split("/")[-1]
    os.makedirs(f"generated/{folder_name}", exist_ok=True)

    base = DiffusionPipeline.from_pretrained("segmind/tiny-sd", torch_dtype=torch.float32).to(device)
    model = load_and_set_lora_ckpt(base, model_name, step_count, device)
    generator = torch.Generator(device=device).manual_seed(67)
    print("Model loaded", folder_name)
    
    for i, prompt in enumerate(prompts):
        image = model(prompt, num_inference_steps=30, generator=generator).images[0]
        image.save(f"generated/{folder_name}/image_{i}.jpeg")

# Run evaluation on each folder
for model_name, _ in models:
    folder_name = model_name.split("/")[-1]
    run_evaluation(
        generated_images_path=f"generated/{folder_name}",
        real_images_path="real_validation/",
        captions=prompts,
        attributes_dict=attributes,
        output_file=f"validation_results/{folder_name}_results.json", 
        compute_bias=False
    )