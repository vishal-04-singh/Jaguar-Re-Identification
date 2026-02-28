#!/usr/bin/env python3
"""
Jaguar Re-Identification Inference Script
==========================================

DESCRIPTION:
    This script loads a trained EfficientNet-B0 + ArcFace model and generates
    pairwise similarity predictions for all 137,270 test image pairs. It
    produces a submission.csv file ready for competition submission.

APPROACH:
    1. Load the best trained model checkpoint (best_model.pth).
    2. Discard the ArcFace classification head — only the backbone + neck
       are used to produce 512-dim embeddings.
    3. Extract embeddings for all 371 test images using Test-Time
       Augmentation (TTA): average the embeddings from the original image
       and its horizontally flipped version, then L2-normalize.
    4. For each of the 137,270 image pairs in test.csv, compute the cosine
       similarity between their embeddings.
    5. Map cosine similarities from [-1, 1] to [0, 1] and write to
       submission.csv.

TEST-TIME AUGMENTATION (TTA):
    Jaguars can appear from either flank, so horizontal flipping captures
    mirror-image spot patterns. Averaging embeddings from both orientations
    produces more robust similarity scores.

SIMILARITY INTERPRETATION:
    - Values close to 1.0 → high confidence the pair shows the SAME jaguar
    - Values close to 0.0 → high confidence the pair shows DIFFERENT jaguars
    - The cosine similarity is mapped from [-1,1] → [0,1] via (sim+1)/2

USAGE:
    python3 inference.py

INPUT:
    - best_model.pth: Trained model checkpoint from train.py
    - test.csv: 137,270 image pairs (query_image, gallery_image)
    - test/test/*.png: 371 test images

OUTPUT:
    - submission.csv: row_id and similarity columns (137,270 rows)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMG_DIR = os.path.join(DATA_DIR, "test", "test")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
MODEL_PATH = os.path.join(DATA_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(DATA_DIR, "submission.csv")

IMG_SIZE = 224
BATCH_SIZE = 64

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ─── Model (must match train.py) ─────────────────────────────────────────────
class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        output = cosine * (1 - one_hot) + target_logits * one_hot
        output *= self.s
        return output


class JaguarReIDModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        backbone_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.backbone_dim = backbone_dim

        self.neck = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.arcface = ArcFaceHead(embedding_dim, num_classes)

    def get_embedding(self, x):
        features = self.backbone(x)
        embedding = self.neck(features)
        return embedding

    def forward(self, x, labels=None):
        embedding = self.get_embedding(x)
        if labels is not None:
            logits = self.arcface(embedding, labels)
            return logits, embedding
        return embedding


# ─── Dataset ──────────────────────────────────────────────────────────────────
class TestImageDataset(Dataset):
    def __init__(self, image_files, img_dir, transform):
        self.image_files = image_files
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, self.image_files[idx]


# ─── TTA (Test-Time Augmentation) ────────────────────────────────────────────
def get_tta_transforms():
    """Return list of transforms for test-time augmentation."""
    base_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return [
        # Original
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            base_norm,
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            base_norm,
        ]),
    ]


def extract_embeddings_tta(model, image_files, img_dir):
    """Extract embeddings with TTA (average original + flipped)."""
    tta_transforms = get_tta_transforms()
    all_embeddings = None

    for t_idx, transform in enumerate(tta_transforms):
        ds = TestImageDataset(image_files, img_dir, transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        embeddings_list = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(DEVICE)
                emb = model.get_embedding(images)
                emb = F.normalize(emb, dim=1)
                embeddings_list.append(emb.cpu())

        embeddings = torch.cat(embeddings_list, dim=0)
        if all_embeddings is None:
            all_embeddings = embeddings
        else:
            all_embeddings = all_embeddings + embeddings
        print(f"  TTA transform {t_idx+1}/{len(tta_transforms)} complete")

    # Average and re-normalize
    all_embeddings = F.normalize(all_embeddings / len(tta_transforms), dim=1)
    return all_embeddings.numpy()


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading model from: {MODEL_PATH}")

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    num_classes = checkpoint["num_classes"]
    embedding_dim = checkpoint["embedding_dim"]
    print(f"  Trained for {checkpoint['epoch']} epochs, Val mAP: {checkpoint['val_map']:.4f}")

    # Build model and load weights
    model = JaguarReIDModel(num_classes, embedding_dim).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Get unique test image filenames
    test_df = pd.read_csv(TEST_CSV)
    test_images = sorted(set(test_df["query_image"].tolist() + test_df["gallery_image"].tolist()))
    print(f"Test images: {len(test_images)}")

    # Extract embeddings with TTA
    print("Extracting embeddings...")
    embeddings = extract_embeddings_tta(model, test_images, TEST_IMG_DIR)
    img_to_idx = {img: i for i, img in enumerate(test_images)}

    # Compute pairwise similarities
    print("Computing pairwise similarities...")
    query_indices = [img_to_idx[q] for q in test_df["query_image"]]
    gallery_indices = [img_to_idx[g] for g in test_df["gallery_image"]]

    query_emb = embeddings[query_indices]
    gallery_emb = embeddings[gallery_indices]

    # Cosine similarity (embeddings are already normalized)
    similarities = np.sum(query_emb * gallery_emb, axis=1)

    # Map from [-1, 1] to [0, 1]
    similarities = (similarities + 1.0) / 2.0
    similarities = np.clip(similarities, 0.0, 1.0)

    # Write submission
    submission = pd.DataFrame({
        "row_id": test_df["row_id"].values,
        "similarity": similarities,
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSubmission saved to: {SUBMISSION_PATH}")
    print(f"  Rows: {len(submission)}")
    print(f"  Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")
    print(f"  Similarity mean: {similarities.mean():.4f}")


if __name__ == "__main__":
    main()
