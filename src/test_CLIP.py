import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
# import supervision
from PIL import Image
from datasets import load_dataset
import cv2
import requests
from io import BytesIO

def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=10)
    response.raise_for_status()  # ensures you notice broken links
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image

ds = load_dataset(
    "OpenFace-CQUPT/HumanCaption-10M",
    split="train[:1]"
)

ex_dp = ds[0]
caption = ex_dp["human_caption"][0]
print(type(caption))
print(f"caption: {caption}")
url = ex_dp["url"]
image = load_image_from_url(url)
image = np.array(image)


# GLOBAL MODEL LOADING
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

example = {"image": image}

def compute_CLIP(image: np.ndarray, text: str) -> float:
    """
    Computes the CLIP scores between an image and a text

    Args:
       image (np.ndarray): an image
       text (str): a text describing the image

    Returns:
        float: the CLIP score for the provided image and text (between 0 and 100)
    """
    # Prepare inputs: convert image and text into tensors, tokenize text, resize image
    inputs = CLIP_PROCESSOR(
        text=[text], images=image, return_tensors="pt", padding=True, truncation=True
    ).to(DEVICE)

    with torch.no_grad():  # turns of gradient tracking
        # Run the CLIP model
        outputs = CLIP_MODEL(**inputs)

        # Extract embeddings
        image_embeddings = outputs.image_embeds
        text_embeddings = outputs.text_embeds

        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # Compute similarity between image and text
        similarity = image_embeddings @ text_embeddings.T

        score = similarity.item()

    final_score = max(score, 0) * 100

    return final_score


def standardize_images(example):
    """
    Standardizes an image by ensuring every image is RGB, NumPY array,
    has consistent size and dtype.

    Args:
        example (dict): example with an image

    Returns:
        dict: the same example with the updated image
    """
    target_size = (224, 224)
    image = example.get("image")

    # convert to RGB
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        image = np.array(image)

    # ensure 3 channels (RGB)
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    # resize
    image = Image.fromarray(image)
    image = image.resize(target_size)

    example["image"] = np.array(image)

    return example

example = standardize_images(example)
score = compute_CLIP(example["image"], caption)
print(score)