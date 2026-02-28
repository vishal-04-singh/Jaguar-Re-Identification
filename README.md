# 🐆 Jaguar Re-Identification

A deep metric learning solution for the **Jaguar Re-Identification** competition. The system identifies individual jaguars from images using an EfficientNet-B0 backbone with ArcFace angular margin loss.

## Architecture

```
Input Image (224×224×3)
  → EfficientNet-B0 Backbone (ImageNet pretrained)
  → Global Average Pooling (1280-dim)
  → Linear + BatchNorm (512-dim embedding)
  → ArcFace Head (31-class angular margin softmax)
```

## Features

- **Backbone**: EfficientNet-B0 with pretrained ImageNet weights
- **Metric Learning**: ArcFace loss (s=30, m=0.5) for discriminative embeddings
- **Data Augmentation**: Random crop, horizontal flip, color jitter, grayscale, random erasing
- **Test-Time Augmentation (TTA)**: Original + horizontal flip embeddings averaged
- **Similarity**: Cosine similarity mapped to [0, 1] for submission
- **Validation**: Identity-balanced mAP (competition metric)

## Project Structure

```
jaguar-re-id/
├── train.py              # Training script
├── inference.py           # Inference & submission generation
├── train.csv              # Training labels (image → jaguar ID)
├── train/train/           # Training images (not included, see below)
├── test/test/             # Test images (not included, see below)
├── test.csv               # Test pairs for submission
├── best_model.pth         # Trained model weights (not included)
├── submission.csv         # Generated submission file
└── sample_submission.csv  # Sample submission format
```

## Setup

### Prerequisites

- Python 3.8+
- PyTorch
- torchvision
- pandas
- scikit-learn
- Pillow
- tqdm

### Install Dependencies

```bash
pip install torch torchvision pandas scikit-learn pillow tqdm
```

### Dataset

> ⚠️ The dataset is **not included** in this repository due to its large size (~16.5 GB).

Download the dataset from the [Kaggle Competition Page](https://www.kaggle.com/competitions/jaguar-re-id/data) and place the files as follows:

```
jaguar-re-id/
├── train/train/*.png      # Training images
├── test/test/*.png        # Test images
├── train.csv
└── test.csv
```

## Usage

### Training

```bash
python3 train.py
```

This will:
1. Load training data with an 80/20 stratified split
2. Train for 10 epochs with AdamW optimizer (lr=3e-4)
3. Compute identity-balanced mAP after each epoch
4. Save the best model to `best_model.pth`

**Device**: Automatically uses MPS (Apple Silicon) → CUDA → CPU.

### Inference

```bash
python3 inference.py
```

This will:
1. Load `best_model.pth`
2. Extract 512-dim embeddings for all test images with TTA
3. Compute pairwise cosine similarity for 137,270 test pairs
4. Save results to `submission.csv`

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image Size | 224×224 |
| Embedding Dim | 512 |
| Batch Size | 32 |
| Learning Rate | 3e-4 |
| Weight Decay | 1e-4 |
| Epochs | 10 |
| ArcFace Scale (s) | 30.0 |
| ArcFace Margin (m) | 0.5 |
| Validation Split | 20% |

## License

This project is for educational and competition purposes.
# -Jaguar-Re-Identification
