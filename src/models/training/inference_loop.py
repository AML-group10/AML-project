import json
import os
import sys

import torch
from datasets import load_dataset
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download
from peft import LoraConfig, LoraModel, set_peft_model_state_dict


def load_and_set_lora_ckpt(pipe, repo_id, step_count, device="cpu"):
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


if __name__ == "__main__":
    # Load validation prompts from HuggingFace
    device = "cpu"
    dataset = load_dataset(
        "AML-group10/AML_project_preprocessed_dataset", "valid", split="train"
    )
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

    # Loop over all 9 models
    models = [("AML-group10/3e-4_10_hyperparameter_tuning", 100)]

    os.makedirs("real_validation", exist_ok=True)
    os.makedirs("validation_results", exist_ok=True)

    """
    for i, item in enumerate(dataset):
        item["image"].save(f"real_validation/image_{i}.jpeg")
    """

    for model_name, step_count in models:
        folder_name = model_name.split("/")[-1]
        os.makedirs(f"generated/{folder_name}", exist_ok=True)

        base = DiffusionPipeline.from_pretrained(
            "segmind/tiny-sd", torch_dtype=torch.float32
        ).to(device)
        model = load_and_set_lora_ckpt(base, model_name, step_count, device)
        generator = torch.Generator(device=device).manual_seed(67)
        print("Model loaded", folder_name)

        for i, prompt in enumerate(prompts):
            image = model(prompt, num_inference_steps=30, generator=generator).images[0]
            image.save(f"generated/{folder_name}/image_{i}.jpeg")
