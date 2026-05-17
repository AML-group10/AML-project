import torch
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from fid_score.fid_score import FidScore
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


    def compute_CLIP(self, image, text):
        inputs = self.processor(
            text=[text],
            images=image,
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

            # score = max(score, 0) * 100

        return score
    

    def compute_CLIP_batch(self, images_dir, texts):
        images, paths = self._load_images_pil(images_dir)

        scores = []

        for image, text in zip(images, texts):
            scores.append(self.compute_CLIP(image, text))
        
        return {
            "mean_CLIP": float(np.mean(scores)),
            "std_CLIP": float(np.std(scores)),
            "scores": scores
        }


    def compute_FID1(self, real_images_dir, gen_images_dir, batch_size = 64):
        """
        real_images_dir: path to the folder with real images
        gen_images_dir: path to the folder with generated images
        """
        paths = [str(real_images_dir), str(gen_images_dir)]
        
        fid = FidScore(paths, self.device, batch_size)
        score = fid.calculate_fid_score()

        print(f"FID Score: {score}")

        return score


    def compute_FID2(self, real_images_dir, gen_images_dir, batch_size=64):
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



evaluator = Evaluator(device="cuda")

# CLIP
clip_results = evaluator.compute_CLIP_batch(
    images_dir="./dog/",
    texts="a photo of TOK dog"
)
print(clip_results)

# FID
fid1 = evaluator.compute_FID1("./dog/", "./generated/")
fid2 = evaluator.compute_FID2("./dog/", "./generated/")
print(f"FID1: {fid1:.2f}  |  FID2: {fid2:.2f}")

# Bias
bias = evaluator.evaluate_bias({
    "with_bucket": [0.31, 0.29, 0.33],
    "without_bucket": [0.21, 0.19, 0.25],
})
print(bias)