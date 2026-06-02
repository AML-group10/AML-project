import json
import os

import torch
from datasets import load_dataset
from diffusers import DiffusionPipeline
from evaluation_functions import run_evaluation
from huggingface_hub import hf_hub_download
from peft import LoraConfig, LoraModel, set_peft_model_state_dict


def load_and_set_lora_ckpt(pipe, repo_id, step_count, device="cpu"):
    """
    Loads the LORA weigths from a HuggingFace repository

    Args:
        pipe (DiffusionPipeline): the loaded base model
        repo_id (str): a HuggingFace repository id where the LORA weights are stored
        step_count (int): the number of steps the model was fine-tuned for
        device (str): the device to which to run the computations
    
    Returns:
        DiffudionPipleline: the base model with loaded LORA weights

    """
    config_path = hf_hub_download(
        repo_id=repo_id, filename=f"{step_count}_lora_config.json"
    )
    weights_path = hf_hub_download(repo_id=repo_id, filename=f"{step_count}_lora.pt")

    with open(config_path, "r") as f:
        lora_config = json.load(f)

    lora_checkpoint_sd = torch.load(weights_path)
    unet_lora_ds = {
        k: v for k, v in lora_checkpoint_sd.items() if "text_encoder_" not in k
    }
    text_encoder_lora_ds = {
        k.replace("text_encoder_", ""): v
        for k, v in lora_checkpoint_sd.items()
        if "text_encoder_" in k
    }

    unet_config = LoraConfig(**lora_config["peft_config"])
    pipe.unet = LoraModel(pipe.unet, unet_config, "default")
    set_peft_model_state_dict(pipe.unet, unet_lora_ds)

    if "text_encoder_peft_config" in lora_config:
        text_encoder_config = LoraConfig(**lora_config["text_encoder_peft_config"])
        pipe.text_encoder = LoraModel(pipe.text_encoder, text_encoder_config, "default")
        set_peft_model_state_dict(pipe.text_encoder, text_encoder_lora_ds)

    pipe.to(device)
    return pipe


# Load validation prompts from HuggingFace
device = "cpu"
dataset = load_dataset(
    "AML-group10/AML_project_preprocessed_dataset", "test", split="train"
)
#dataset = dataset.shuffle(seed=42).select(range(700))
prompts = [item["prompt"][0] for item in dataset]
#prompts = prompts[0:256]

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

"""
model_name = "AML-group10/5e-4_20_hyperparameter_tuning"
base = DiffusionPipeline.from_pretrained(
    "segmind/tiny-sd", torch_dtype=torch.float32
).to(device)

os.makedirs("real_test", exist_ok=True)
os.makedirs("test_results", exist_ok=True)
"""

# Save true images
for i, item in enumerate(dataset):
    item["image"].save(f"real_test/image_{i}.jpeg")

"""
# generate images from the model and run evaluation
folder_name = model_name.split("/")[-1]
os.makedirs(f"generated_test/{folder_name}", exist_ok=True)
model = load_and_set_lora_ckpt(base, model_name, 200, device)
generator = torch.Generator(device=device).manual_seed(67)
print("Model loaded", folder_name)

for i, prompt in enumerate(prompts):
    image = model(prompt, num_inference_steps=30, generator=generator).images[0]
    image.save(f"generated_test/{folder_name}/image_{i}.jpeg")

run_evaluation(
    generated_images_path=f"generated_test/{folder_name}",
    real_images_path="real_test/",
    captions=prompts,
    attributes_dict=attributes,
    output_file=f"test_results/{folder_name}_results.json",
    compute_bias=True,
)



# generate images from the baseline model and run evaluation
folder_name = "baseline"
os.makedirs(f"generated_test/{folder_name}", exist_ok=True)
generator = torch.Generator(device=device).manual_seed(67)

for i, prompt in enumerate(prompts):
    if i in range(0,868):
        continue
    image = base(prompt, num_inference_steps=30, generator=generator).images[0]
    image.save(f"generated_test/{folder_name}/image_{i}.jpeg")
"""

folder_name = "baseline"
run_evaluation(
    generated_images_path=f"generated_test/{folder_name}",
    real_images_path="real_test/",
    captions=prompts,
    attributes_dict=attributes,
    output_file=f"test_results/{folder_name}_results.json",
    compute_bias=True,
)
