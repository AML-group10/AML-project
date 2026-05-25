from datasets import load_dataset
import os
from evaluation_functions import run_evaluation
from models.training.inference import load_and_set_lora_ckpt

# Load validation prompts from HuggingFace
dataset = load_dataset("AML-group10/AML_project_preprocessed_dataset", split="valid")
prompts = [item["prompt"] for item in dataset]

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

# Loop over all 9 models
models = [
    ("AML-group10/1e-4_20hyperparameter_tuning", 200),
    ("AML-group10/1e-4_15hyperparameter_tuning", 150),
    ("AML-group10/1e-4_10hyperparameter_tuning", 100),
    ("AML-group10/5e-4_10hyperparameter_tuning", 100),
    ("AML-group10/5e-4_15hyperparameter_tuning", 150),
    ("AML-group10/5e-4_20hyperparameter_tuning", 200),
    ("AML-group10/3e-4_10hyperparameter_tuning", 100),
    ("AML-group10/3e-4_15hyperparameter_tuning", 150),
    ("AML-group10/3e-4_20hyperparameter_tuning", 200)
]

os.makedirs("validation_results", exist_ok=True)

for model_name, step_count in models:
    folder_name = model_name.split("/")[-1]
    os.makedirs(f"generated/{folder_name}", exist_ok=True)

    base = DiffusionPipeline.from_pretrained("segmind/tiny-sd", torch_dtype=torch.float32)
    model = load_and_set_lora_ckpt(base, step_count)  
    print("Model loaded", model_name)
    
    for i, prompt in enumerate(prompts):
        generator = torch.Generator(device="cpu").manual_seed(67)
        image = model(prompt, num_inference_steps=30, generator=generator).images[0]
        image.save(f"generated/{folder_name}/image_{i}.jpeg")

# Run evaluation on each folder
for model_name in models:
    folder_name = model_name.split("/")[-1]
    run_evaluation(
        generated_images_path=f"generated/{folder_name}",
        real_images_path="real_validation/",
        captions=prompts,
        attributes_dict=attributes,
        output_file=f"validation_results/{folder_name}_results.json", 
        compute_bias=False
    )