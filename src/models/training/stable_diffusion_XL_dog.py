#   Run these shell commands before executing the Python code below:
#
#   pip install bitsandbytes transformers torchao==0.16.0 accelerate peft -q
#   pip install git+https://github.com/huggingface/diffusers.git -q
#   pip install datasets -q
#   wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_sdxl.py
#   accelerate config default

""" Dataset """

from huggingface_hub import snapshot_download
from PIL import Image
import glob
import requests
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch
import json
import gc
from huggingface_hub import login
import subprocess
from huggingface_hub import whoami, create_repo, upload_folder
from pathlib import Path
from diffusers import DiffusionPipeline, AutoencoderKL

local_dir = "./dog/"
snapshot_download(
    "diffusers/dog-example",
    local_dir=local_dir, repo_type="dataset",
    ignore_patterns=".gitattributes",
)

"""Preview the images:"""

def image_grid(imgs, rows, cols, resize=256):
    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


# change path to display images from your local dir
img_paths = "./dog/*.jpeg"
imgs = [Image.open(path) for path in glob.glob(img_paths)]
num_imgs_to_preview = 5
grid = image_grid(imgs[:num_imgs_to_preview], 1, num_imgs_to_preview)
grid.save("preview.png")

"""### Generate custom captions with BLIP
Load BLIP to auto caption your images:
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

# load the processor and the captioning model
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",torch_dtype=torch.float16).to(device)

# captioning utility
def caption_images(input_image):
    inputs = blip_processor(images=input_image, return_tensors="pt").to(device, torch.float16)
    pixel_values = inputs.pixel_values

    generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption

# create a list of (Pil.Image, path) pairs
local_dir = "./dog/"
imgs_and_paths = [(path,Image.open(path)) for path in glob.glob(f"{local_dir}*.jpeg")]

caption_prefix = "a photo of TOK dog, " #@param
with open(f'{local_dir}metadata.jsonl', 'w') as outfile:
  for img in imgs_and_paths:
      caption = caption_prefix + caption_images(img[1]).split("\n")[0]
      entry = {"file_name":img[0].split("/")[-1], "prompt": caption}
      json.dump(entry, outfile)
      outfile.write('\n')

"""Free some memory:"""

# delete the BLIP pipelines and free up some memory
del blip_processor, blip_model
gc.collect()
torch.cuda.empty_cache()

login(token="put your token here")

""" Train """

train_cmd = [
    "accelerate", "launch", "train_dreambooth_lora_sdxl.py",
    "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-xl-base-1.0",
    "--pretrained_vae_model_name_or_path", "madebyollin/sdxl-vae-fp16-fix",
    "--instance_data_dir", "dog",
    "--output_dir", "corgy_dog_LoRA",
    "--caption_column", "text",
    "--mixed_precision", "fp16",
    "--instance_prompt", "a photo of TOK dog",
    "--resolution", "1024",
    "--train_batch_size", "1",
    "--gradient_accumulation_steps", "3",
    "--gradient_checkpointing",
    "--learning_rate", "1e-4",
    "--snr_gamma", "5.0",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--use_8bit_adam",
    "--max_train_steps", "500",
    "--checkpointing_steps", "717",
    "--seed", "0",
    "--cast_teacher_unet",]

subprocess.run(train_cmd, check=True)

### Save your model to the hub and check it out """

from train_dreambooth_lora_sdxl import save_model_card

output_dir = "corgy_dog_LoRA" #@param
username = whoami(token=Path("/root/.cache/huggingface/token"))["name"]
repo_id = f"{username}/{output_dir}"

repo_id = create_repo(repo_id, exist_ok=True).repo_id

# change the params below according to your training arguments
save_model_card(
    repo_id = repo_id,
    images=[],
    base_model="stabilityai/stable-diffusion-xl-base-1.0",
    train_text_encoder=False,
    instance_prompt="a photo of TOK dog",
    validation_prompt=None,
    repo_folder=output_dir,
    vae_path="madebyollin/sdxl-vae-fp16-fix",
    use_dora =False,
)

upload_folder(
    repo_id=repo_id,
    folder_path=output_dir,
    commit_message="End of training",
    ignore_patterns=["step_*", "epoch_*"],
)


"""Let's generate some images with it!

## Inference
"""

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.load_lora_weights(repo_id)
_ = pipe.to("cuda")

prompt = "a photo of husky dog in a pool" 

image = pipe(prompt=prompt, num_inference_steps=25).images[0]
image.save("output.png")
print("Image saved to output.png")