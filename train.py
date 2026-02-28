#!/usr/bin/env python3
"""
Jaguar Re-Identification Training Script
=========================================

DESCRIPTION:
    This script trains a deep metric learning model to distinguish between
    31 individual jaguars based on their unique spot patterns. It uses an
    ArcFace (Additive Angular Margin) loss function combined with an
    EfficientNet-B0 backbone pretrained on ImageNet.

APPROACH:
    1. Backbone: EfficientNet-B0 (torchvision, ImageNet-pretrained) extracts
       visual features from 224x224 jaguar images.
    2. Neck: A linear projection layer maps backbone features to a 512-dim
       embedding space, followed by BatchNorm for training stability.
    3. ArcFace Head: Applies an angular margin penalty to the softmax loss,
       forcing the model to learn more discriminative embeddings. This is
       the standard approach for face/animal re-identification tasks.
    4. Class-Balanced Sampling: A WeightedRandomSampler ensures that
       under-represented jaguars (as few as 13 images) are sampled
       proportionally to over-represented ones (up to 183 images).
    5. Data Augmentation: RandomResizedCrop, horizontal flip, color jitter,
       grayscale, and random erasing to improve generalization.
    6. Validation: An 80/20 stratified split is used to compute
       identity-balanced mAP (the competition metric) after each epoch.

ARCHITECTURE:
    Input Image (224x224x3)
        → EfficientNet-B0 Backbone (pretrained)
        → Global Average Pooling (1280-dim)
        → Linear + BatchNorm (512-dim embedding)
        → ArcFace Head (31-class angular margin softmax)

USAGE:
    python3 train.py

OUTPUT:
    - best_model.pth: Model checkpoint with highest validation mAP
    - Console logs: Per-epoch loss, accuracy, validation mAP, and timing

HYPERPARAMETERS:
    - Embedding dim: 512
    - ArcFace scale (s): 30.0, margin (m): 0.5
    - Learning rate: 3e-4 with cosine annealing
    - Optimizer: AdamW (weight decay 1e-4)
    - Batch size: 32, Epochs: 10
    - Image size: 224x224

DATASET:
    - 1,896 training images of 31 jaguars (train/train/*.png)
    - Labels from train.csv (filename → jaguar name)
    - Stratified 80/20 train/val split
"""

import os
import math
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "train")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "best_model.pth")

EMBEDDING_DIM = 512
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
SEED = 42
IMG_SIZE = 224

# ArcFace hyperparams
ARCFACE_S = 30.0
ARCFACE_M = 0.5

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ─── Dataset ─────────────────────────────────────────────────────────────────
class JaguarDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = row["label_id"]
        return image, label


# ─── ArcFace Head ─────────────────────────────────────────────────────────────
class ArcFaceHead(nn.Module):
    """Additive Angular Margin Loss for deep face/re-id recognition."""

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


# ─── Model ────────────────────────────────────────────────────────────────────
class JaguarReIDModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super().__init__()
        # Use EfficientNet-B0 from torchvision (small, fast, good accuracy)
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b0(weights=weights)
        # Remove classifier head
        backbone_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.backbone_dim = backbone_dim

        self.neck = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.arcface = ArcFaceHead(embedding_dim, num_classes, s=ARCFACE_S, m=ARCFACE_M)

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


# ─── Evaluation ───────────────────────────────────────────────────────────────
def compute_identity_balanced_map(embeddings, labels):
    """Compute identity-balanced mAP from embeddings."""
    embeddings = F.normalize(torch.tensor(embeddings), dim=1).numpy()
    sim_matrix = embeddings @ embeddings.T

    unique_labels = np.unique(labels)
    ap_per_identity = {}

    for identity in unique_labels:
        identity_indices = np.where(labels == identity)[0]
        aps = []

        for query_idx in identity_indices:
            sims = sim_matrix[query_idx].copy()
            sims[query_idx] = -1  # exclude self

            sorted_indices = np.argsort(-sims)
            gt = (labels[sorted_indices] == identity).astype(float)

            if gt.sum() == 0:
                continue
            cumsum = np.cumsum(gt)
            precision = cumsum / (np.arange(len(gt)) + 1)
            ap = (precision * gt).sum() / gt.sum()
            aps.append(ap)

        if aps:
            ap_per_identity[identity] = np.mean(aps)

    return np.mean(list(ap_per_identity.values()))


# ─── Training ─────────────────────────────────────────────────────────────────
def train():
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Starting training at {time.strftime('%H:%M:%S')}")

    # Load data
    df = pd.read_csv(TRAIN_CSV)
    label_map = {name: i for i, name in enumerate(sorted(df["ground_truth"].unique()))}
    df["label_id"] = df["ground_truth"].map(label_map)
    num_classes = len(label_map)
    print(f"Classes: {num_classes}, Total images: {len(df)}")

    # Stratified split
    train_dfs, val_dfs = [], []
    for label in df["label_id"].unique():
        sub = df[df["label_id"] == label]
        sub = sub.sample(frac=1, random_state=SEED)
        split = max(1, int(len(sub) * (1 - VAL_SPLIT)))
        train_dfs.append(sub.iloc[:split])
        val_dfs.append(sub.iloc[split:])
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Augmentations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = JaguarDataset(train_df, TRAIN_IMG_DIR, train_transform)
    val_ds = JaguarDataset(val_df, TRAIN_IMG_DIR, val_transform)

    # Class-balanced sampler
    class_counts = train_df["label_id"].value_counts().sort_index().values
    sample_weights = 1.0 / class_counts[train_df["label_id"].values]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)

    # Model
    print("Building model...")
    model = JaguarReIDModel(num_classes, EMBEDDING_DIM).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_map = 0.0
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        # ── Train ──
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits, _ = model(images, labels)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total
        scheduler.step()

        # ── Validate ──
        model.eval()
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                embeddings = model.get_embedding(images)
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.numpy())

        all_embeddings = np.concatenate(all_embeddings)
        all_labels = np.concatenate(all_labels)

        val_map = compute_identity_balanced_map(all_embeddings, all_labels)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Acc: {train_acc:.4f} | Val mAP: {val_map:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | Time: {epoch_time:.1f}s")

        # Save best
        if val_map > best_map:
            best_map = val_map
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_map": label_map,
                "embedding_dim": EMBEDDING_DIM,
                "backbone_dim": model.backbone_dim,
                "num_classes": num_classes,
                "val_map": val_map,
                "epoch": epoch + 1,
            }, MODEL_SAVE_PATH)
            print(f"  ✓ Saved best model (mAP: {val_map:.4f})")

    print(f"\nTraining complete! Best Val mAP: {best_map:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
