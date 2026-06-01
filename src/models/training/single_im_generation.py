import argparse
import json
import os

import torch
from diffusers import DiffusionPipeline
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


def parse_args():
    """
    Parses the arguments needed to generate an image from a prompt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="image.jpeg")
    return parser.parse_args()


args = parse_args()
prompt = args.prompt
output_path = os.path.join("generated/single_prompts", args.output)

# Load validation prompts from HuggingFace
device = "cpu"

model_name = "AML-group10/5e-4_20_hyperparameter_tuning"
step_count = 200

os.makedirs("validation_results", exist_ok=True)

os.makedirs("generated/single_prompts", exist_ok=True)

base = DiffusionPipeline.from_pretrained(
    "segmind/tiny-sd", torch_dtype=torch.float32
).to(device)
model = load_and_set_lora_ckpt(base, model_name, step_count, device)
generator = None
print("Model loaded")


image = model(prompt, num_inference_steps=30, generator=generator).images[0]
image.save(output_path)

print(f"Saved image in {output_path}")
