from datasets import load_dataset
import os
from evaluation_functions import run_evaluation

# Load validation prompts from HuggingFace
dataset = load_dataset("AML-group10/AML_project_preprocessed_dataset", "valid")
prompts = [item["prompt"] for item in dataset["train"]]

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
    "AML-group10/1e-4_20hyperparameter_tuning",
    "AML-group10/1e-4_15hyperparameter_tuning",
    "AML-group10/1e-4_10hyperparameter_tuning",
    "AML-group10/5e-4_10hyperparameter_tuning",
    "AML-group10/5e-4_15hyperparameter_tuning",
    "AML-group10/5e-4_20hyperparameter_tuning",
    "AML-group10/3e-4_10hyperparameter_tuning",
    "AML-group10/3e-4_15hyperparameter_tuning",
    "AML-group10/3e-4_20hyperparameter_tuning",
]

for model_name in models:
    # Create output folder for this model
    folder_name = model_name.split("/")[-1] 
    os.makedirs(f"generated/{folder_name}", exist_ok=True)
    
    # Loop over all validation prompts
    for i, prompt in enumerate(prompts):
        pass
        # GENERATE IMAGE AND SAVE IT IN THE FOLDER

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