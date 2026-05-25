from diffusers import DiffusionPipeline
import torch
import os
import json
from peft import LoraConfig, LoraModel, set_peft_model_state_dict

def load_and_set_lora_ckpt(pipe, step_count):
    with open(os.path.join("./AML-group10/lora-output", f"{step_count}_lora_config.json"), "r") as f:
        lora_config = json.load(f)
    print(lora_config)

    lora_checkpoint_sd = torch.load(f"./AML-group10/lora-output/{step_count}_lora.pt")
    unet_lora_ds = {k: v for k, v in lora_checkpoint_sd.items() if "text_encoder_" not in k}
    text_encoder_lora_ds = {
        k.replace("text_encoder_", ""): v for k, v in lora_checkpoint_sd.items() if "text_encoder_" in k
    }

    unet_config = LoraConfig(**lora_config["peft_config"])
    pipe.unet = LoraModel(pipe.unet, unet_config, "default")
    set_peft_model_state_dict(pipe.unet, unet_lora_ds)

    if "text_encoder_peft_config" in lora_config:
        text_encoder_config = LoraConfig(**lora_config["text_encoder_peft_config"])
        pipe.text_encoder = LoraModel(pipe.text_encoder, text_encoder_config, "default")
        set_peft_model_state_dict(pipe.text_encoder, text_encoder_lora_ds)

    pipe.to("cpu")
    return pipe

base = DiffusionPipeline.from_pretrained("segmind/tiny-sd", torch_dtype=torch.float32)
model = load_and_set_lora_ckpt(base, 10)

print("Model loaded")

prompt = "Portrait of a pretty girl"
generator = torch.Generator(device="cpu").manual_seed(67)
image = model(prompt, num_inference_steps=30, generator=generator).images[0]
image.save("image_model.jpeg")

#base model
image = base(prompt).images[0]
image.save("image_base.jpeg")