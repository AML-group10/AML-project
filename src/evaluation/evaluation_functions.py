import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
from PIL import Image
import pandas as pd
import re
import json

# The file allows for evaluation of image generation quality and bias using CLIP and FID scores.
# - CLIP score: measures how well a generated images matches the text caption
# - FID score: measues how similar the distribution of generated images is to real images
# - bias table: compares CLIP scores across attribute groups based on image captions

class ImageFolderDataset(Dataset):
  """
  A PyTorch dataset that loads images from a folder on disk. All images are converted to RGB on load.
  """
  def __init__(self, folder_path, transform=None):
        EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.paths = [p for p in Path(folder_path).iterdir() if p.suffix.lower() in EXTENSIONS]
        self.transform = transform

  def __len__(self):
        return len(self.paths)

  def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
  

class Evaluator:
    """
    Evaluates image generation quality and bias using CLIP and FID.
    """

    def __init__(self, device="cuda"):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        self.fid_transform = T.Compose([T.Resize((299, 299)), T.ToTensor(), T.Lambda(lambda x: (x * 255).byte())])


    def _load_images_pil(self, folder_path: str) -> tuple[list, list[str]]:
        """
        Load all images from a folder as PIL Images.
        Args:
            folder_path (str): path to the folder containing images
        Returns:
            Tuple of (list of PIL Images, list of file path strings)
        """
        dataset = ImageFolderDataset(folder_path)
        return [dataset[i] for i in range(len(dataset))], [str(p) for p in dataset.paths]


    def _load_images_tensor(self, folder_path, batch_size=64):
        """
        Load a folder of images into a single batched tensor for FID.
        Args:
            folder_path: path to the folder containing images
            batch_size: number of images to process at a time
        Returns:
            Tensor of shape (N, 3, 299, 299) with dtype uint8
        """
        dataset = ImageFolderDataset(folder_path, transform=self.fid_transform)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
  
        return torch.cat([batch for batch in loader])


    def _compute_CLIP(self, image: Image.Image, text: str) -> float:
        """
        Args:
            image: a PIL Image 
            text: prompt/caption string
        Returns:
            float: roughly between -1 and 1
        """
        inputs = self.processor(
            text=[text],
            images=image.convert("RGB"),
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad(): # turns of gradient tracking
            # Run the CLIP model
            outputs = self.clip_model(**inputs)

            # Extract embeddings
            image_embeddings = outputs.image_embeds
            text_embeddings = outputs.text_embeds

            # Normalize embeddings
            image_embeddings = F.normalize(image_embeddings, dim=-1)
            text_embeddings = F.normalize(text_embeddings, dim=-1)

            # Compute similarity between image and text
            similarity = image_embeddings @ text_embeddings.T

            score = similarity.item()

        return score
    

    def compute_CLIP_batch(self, images_dir: str, texts: list[str]) -> dict:
        """
        Compute CLIP scores for a batch of images and captions.

        Args:
            images_dir (str): Folder path
            texts (list[str]): List of captions
        """
        if isinstance(images_dir, (str, Path)):
            images, _ = self._load_images_pil(images_dir)
        else:
            raise TypeError("'images_dir must be a folder path")
        
        if len(texts) != len(images):
            raise ValueError(f"Got {len(images)} images, but {len(texts)} texts")

        scores = []

        for image, text in zip(images, texts):
            scores.append(self._compute_CLIP(image, text))
        
        return {
            "mean_CLIP": float(np.mean(scores)),
            "std_CLIP": float(np.std(scores)),
            "scores": scores
        }

    def compute_FID(self, real_images_dir: str, gen_images_dir: str, batch_size: int =64) -> float:
        """
        Compute the Frechet Inception Distance between two folders of images.

        Args:
            real_images_dir: path to the folder with real images
            gen_images_dir: path to the folder with generated images
            batch_size: number of images to process at a time

        Returns:
            FID score as a flot, the lower the better
        """
        fid = FrechetInceptionDistance(feature=2048).to(self.device)
        fid.update(self._load_images_tensor(real_images_dir, batch_size).to(self.device), real = True)
        fid.update(self._load_images_tensor(gen_images_dir, batch_size).to(self.device), real = False)
        
        score = fid.compute().item()

        return score

    def _keyword_in_caption(self, kw: str, caption: str) -> bool:
        """
        Checks whether a keyword appears as a whole word in a caption.
        """
        return bool(re.search(rf"\b{re.escape(kw)}\b", caption.lower()))

    def compute_attribute_clip_table(self, images, captions: list[str], attributes: dict) -> pd.DataFrame:
        """
        Compute a CLIP score comparison table across attribute groups.

        Args:
            images (str): folder path string
            captions (list[str]): 
            attributes (dict): example is {group: [keywords]}
        Returns:
            pd.DataFrame: table with columns Group | Mean CLIP | Std CLIP
        """
        pil_images, _ = self._load_images_pil(images)
        rows= []

        if len(pil_images) != len(captions):
            raise ValueError(f"Got {len(pil_images)} images but {len(captions)} captions")

        for group, keywords in attributes.items():
            # filter images whose caption contains any keyword from this group
            matched = [
                (image, caption) for image, caption in zip(pil_images, captions)
                if any(self._keyword_in_caption(kw, caption) for kw in keywords)
            ]
            print(f"matched: {matched}")

            if not matched:
                rows.append({"Group": group, "Number of images": 0, "Mean CLIP": None, "Std CLIP": None})
                continue

            matched_images, matched_captions = zip(*matched)
            scores = [self._compute_CLIP(img, cap) for img, cap in zip(matched_images, matched_captions)]

            rows.append({
                "Group":      group,
                "Number of images": len(scores),
                "Mean CLIP":  round(float(np.mean(scores)), 4),
                "Std CLIP":   round(float(np.std(scores)),  4),
            })

        return pd.DataFrame(rows)


def run_evaluation(
        generated_images_path: str, 
        real_images_path: str, 
        captions: list[str], 
        attributes_dict: dict, 
        output_file: str, 
        compute_bias: bool):
    """
    Run a full evaluation pipeline on a set of generated images. Computes three metrics:
    1. CLIP score
    2. FID score
    3. Bias table

    Args:
        generated_images_path (str): path to the folder containing generated images
        real_images_path (str): path to the folder containing real images
        captions (list[str]): list of caption strings, one per generated image in the same order
            as the images in the folder
        attributes_dict (dict): Dict mapping group names to lists of keywords
        output_file (str): Name of the file where the results will be saved
        compute_bias (bool): True if the function should calculate bias table
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = Evaluator(device=device)

    # CLIP scores
    clip_results = evaluator.compute_CLIP_batch(
        images_dir=generated_images_path,
        texts=captions)
    print(f"CLIP scores {clip_results}")

    # FID scores
    fid_score = evaluator.compute_FID(real_images_path, generated_images_path)
    print(f"FID2: {fid_score:.2f}")

    # CLIP bias
    if compute_bias:
        bias_df = evaluator.compute_attribute_clip_table(generated_images_path, captions, attributes_dict)
        print(bias_df.to_string(index=False))

    if output_file:
        results = {
            "model": generated_images_path,
            "clip_mean": clip_results["mean_CLIP"],
            "clip_std": clip_results["std_CLIP"],
            "clip_all_scores:": clip_results["scores"],
            "fid": fid_score,
            "bias_table": bias_df.to_dict(orient="records") if compute_bias else {}
        }

        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {output_file}")
