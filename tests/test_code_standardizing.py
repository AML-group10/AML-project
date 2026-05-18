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


# GLOBAL MODEL LOADING
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def standardize_images(example):
    """
    Standardizes an image by ensuring every image is RGB, NumPY array,
    has consistent size and dtype.

    Args:
        example (dict): example with an image

    Returns:
        dict: the same example with the updated image
    """
    target_size = (512, 512)
    image = example.get("image")


    # if numpy array from OpenCV -> convert BGR to RGB
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image[..., ::-1]   # BGR -> RGB

    # PIL conversion
    image = Image.fromarray(image).convert("RGB")

    # grayscale handling
    image = np.array(image)
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    # ensure 3 channels (RGB)
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    # resize
    image = Image.fromarray(image)
    image = image.resize(target_size)

    example["image"] = np.array(image)

    return example

for i, data_point in enumerate(ds):
    ex_dp = ds[0]
    caption = ex_dp["human_caption"]
    print(type(caption))
    print(f"caption: {caption}")
    url = ex_dp["url"]
    image = load_image_from_url(url)
    image = np.array(image)
    example = {"image": image}
    example = standardize_images(example)
    cv2.imwrite(f"example{i}.jpg", example["image"])