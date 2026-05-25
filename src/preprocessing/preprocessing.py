from datasets import Dataset, DatasetDict
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
# import supervision
from PIL import Image

# GLOBAL MODEL LOADING
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
YOLO_PATH = hf_hub_download(
        repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt"
    )

YOLO_MODEL = YOLO(YOLO_PATH)

def preprocess(dataset: Dataset) -> Dataset:
    """
    Preprocesses the data in a dataset

    Args:
        dataset (Dataset): the dataset to be preprocessed

    Returns:
        Dataset: the preprocessed dataset
    """
    dataset = _remove_and_rename_features(dataset)
    dataset = _remove_unwanted_samples(dataset)
    dataset = _preprocess_captions(dataset)
    dataset = _preprocess_images(dataset)
    dataset =  _split_data(dataset)
    return dataset


def _remove_and_rename_features(dataset: Dataset) -> Dataset:
    """
    Removes the columns that will not be used in the training of the model and renames the existing columns with suitable names

    Args:
        dataset (Dataset): the dataset with the old columns

    Returns:
        Dataset: the dataset with removed columns and suitable names
    """
    dataset = dataset.remove_columns(["image_id", "image", "laion_caption", "sha256"])
    return dataset.rename_column("url", "image").rename_column(
        "human_caption", "prompt"
    )


def _remove_unwanted_samples(dataset: DatasetDict) -> DatasetDict:
    """
    Removes unwanted samples (null values, nudity, multiple people, low CLIP scores) from a dataset

    Args:
        dataset (Dataset): the dataset with all samples

    Returns:
        Dataset: the dataset with removed unwanted samples
    """
    # remove null values
    dataset = dataset.filter(lambda example: example["image"] is not None)

    # remove nudity and multiple people
    dataset = dataset.filter(
        lambda example: not any(
            [
                word in example["prompt"]
                for word in [
                    "naked",
                    "nude",
                    "breasts",
                    "undressed",
                    "topless",
                ]
            ]
        )
    )

    # remove low CLIP scores
    dataset = dataset.filter(
        lambda example: _compute_CLIP(example["image"], example["prompt"][0]) > 20
    )

    return dataset


def _preprocess_captions(dataset: Dataset) -> Dataset:
    """
    Preprocesses the captions in a dataset

    Args:
        dataset (Dataset): dataset to be preprocessed

    Returns:
        Dataset: dataset with preprocessed captions
    """
    return dataset.map(_preprocess_caption)


def _preprocess_images(dataset: Dataset) -> Dataset:
    """
    Preprocesses the images in a dataset

    Args:
        dataset (Dataset): dataset to be preprocessed

    Returns:
        Dataset: dataset with preprocessed images
    """
    # crop images
    dataset = dataset.map(_crop_images)
    dataset = dataset.filter(lambda example: example["image"] is not None)
    # standardize images
    dataset = dataset.map(_standardize_images)
    return dataset


def _preprocess_caption(example: dict) -> dict:
    """
    Preprocesses a single caption

    Args:
        example (dict): example with a caption

    Returns:
        dict: the same example with the augmented caption
    """
    # lowercase
    caption = example["prompt"].lower().strip(" ", 5)
    # cut out first 5 words
    words = caption.split()
    if len(words) == 6: # I included that in case there are very short captions
        caption = words[5]
    # set max length
    example["prompt"] = [caption]
    return example


def _split_data(dataset: Dataset) -> Dataset:
    """
    Spltits the data into train, validation, and test datasets

    Args:
        dataset (Dataset): the dataset to be split

    Returns:
        Dataset: a dataset split into train, validation, and test datasets
    """
    # 70% train, 30% test + validation
    train_valtest = dataset.train_test_split(test_size=0.3, seed=42)
    # Split the 30% test in half for validation and half for testing
    val_test = train_valtest["test"].train_test_split(test_size=0.5, shuffle=False)

    train_test_valid_dataset = DatasetDict(
        {
            "train": train_valtest["train"],
            "valid": val_test["train"],
            "test": val_test["test"],
        }
    )
    return train_test_valid_dataset


def _compute_CLIP(image: np.ndarray, text: str) -> float:
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


def _crop_images(example: dict) -> dict:
    """
    Crops an image with a face centered

    Args:
        example (dict): example with an image

    Returns:
        dict: the same example with the updated image
    """
    image = Image.fromarray(np.array(example["image"]).astype("uint8"))
    results = YOLO_MODEL(image)[0]
    # detections = supervision.Detections.from_ultralytics(results)
    h, w = np.array(example["image"]).shape[:2]
    boxes = results.boxes.xyxy.cpu().numpy()

    if len(boxes) != 1:
        example["image"] = None
        return example

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # box width/height
        bw = x2 - x1
        bh = y2 - y1

        # padding
        pad_x = int(0.5 * bw)
        pad_y_down = int(0.3 * bh)
        pad_y_up = int(0.5 * bh)

        # expand box safely
        x1_new = max(0, x1 - pad_x)
        y1_new = max(0, y1 - pad_y_up)
        x2_new = min(w, x2 + pad_x)
        y2_new = min(h, y2 + pad_y_down)

        example["image"] = np.array(example["image"])[y1_new:y2_new, x1_new:x2_new]

    return example

def _standardize_images(example):
    """
    Standardizes an image by ensuring every image is RGB, NumPY array,
    has consistent size and dtype.

    Args:
        example (dict): example with an image

    Returns:
        dict: the same example with the updated image
    """
    target_size = (768, 768)
    image = np.array(example["image"]).astype("uint8")

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

    # resize
    image = Image.fromarray(image)
    image = image.resize(target_size)

    example["image"] = np.array(image)

    return example