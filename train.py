import re
from pathlib import Path

import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path("data/NEU-DET")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
EPOCHS = 10
BATCH_SIZE = 32
LR = 3e-4
IMG_SIZE = 224

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class NEUDataset(Dataset):
    def __init__(self, split, tf):
        self.files = sorted((ROOT / split / "images").rglob("*.jpg"))
        self.tf = tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        label = CLASSES.index(re.sub(r"_\d+$", "", f.stem))
        img = Image.open(f).convert("RGB")  # NEU images are grayscale; model expects 3 channels
        return self.tf(img), label


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds += out.argmax(1).cpu().tolist()
            labels += y.tolist()
    return labels, preds
