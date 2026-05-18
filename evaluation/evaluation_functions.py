import torch
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
from PIL import Image

class ImageFolderDataset(Dataset):
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
    def __init__(self, device="cuda"):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        self.fid_transform = T.Compose([T.Resize((299, 299)), T.ToTensor(), T.Lambda(lambda x: (x * 255).byte())])


    def _load_images_pil(self, folder_path):
        dataset = ImageFolderDataset(folder_path)
        return [dataset[i] for i in range(len(dataset))], [str(p) for p in dataset.paths]


    def _load_images_tensor(self, folder_path, batch_size=64):
        dataset = ImageFolderDataset(folder_path, transform=self.fid_transform)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
  
        return torch.cat([batch for batch in loader])


    def compute_CLIP(self, image: Image.Image, text: str) -> float:
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
            scores.append(self.compute_CLIP(image, text))
        
        return {
            "mean_CLIP": float(np.mean(scores)),
            "std_CLIP": float(np.std(scores)),
            "scores": scores
        }

    def compute_FID(self, real_images_dir: str, gen_images_dir: str, batch_size: int =64) -> float:
        """
        real_images_dir: path to the folder with real images
        gen_images_dir: path to the folder with generated images
        """
        fid = FrechetInceptionDistance(feature=2048).to(self.device)
        fid.update(self._load_images_tensor(real_images_dir, batch_size).to(self.device), real = True)
        fid.update(self._load_images_tensor(gen_images_dir, batch_size).to(self.device), real = False)
        
        score = fid.compute().item()

        return score


    def evaluate_bias(self, group_scores):
        """
        Each group has a list of scores. The scores can be CLIP scores, FID scores, human ratings etc.
        The function computes mean scores within the groups and the variance between groups to measure
        how different the group performances are.

        Args:
            group_scores (dict(str : list(float))): Example - {"male": [0.5, 0.2, 0.1], "female": [0.9, 0.8, 0.4]}
        """
        group_means = {group: np.mean(scores) for group, scores in group_scores.items()}

        values = list(group_means.values())

        bias_score = np.var(values)

        return {"bias_score": float(bias_score), "group_means": group_means}


def test_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = Evaluator(device=device)

    # CLIP
    clip_results = evaluator.compute_CLIP_batch(
        images_dir="evaluation/gen_test/",
        texts=["A woman with brown mediun length hair", "A man with short dark hair and short beard", "A man with red t-shirt"]
    )
    print(f"CLIP scores {clip_results}")

    # FID
    fid2 = evaluator.compute_FID("evaluation/real_test/", "evaluation/gen_test/")
    print(f"FID2: {fid2:.2f}")

    # Bias
    bias = evaluator.evaluate_bias({
        "with_bucket": [0.31, 0.29, 0.33],
        "without_bucket": [0.21, 0.19, 0.25],
    })
    print(bias)

test_evaluation()