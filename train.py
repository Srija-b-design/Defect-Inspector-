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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # num_workers=0 avoids DataLoader problems on Windows
    train_loader = DataLoader(NEUDataset("train", train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(NEUDataset("validation", val_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        scheduler.step()

        labels, preds = evaluate(model, val_loader, device)
        acc = sum(l == p for l, p in zip(labels, preds)) / len(labels)
        print(f"Epoch {epoch}/{EPOCHS}  train loss {total_loss / len(train_loader.dataset):.4f}  val acc {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "model.pth")
            print("  saved model.pth")

    # Final report using the best saved model
    model.load_state_dict(torch.load("model.pth", map_location=device))
    labels, preds = evaluate(model, val_loader, device)
    print("\nBest val accuracy:", round(best_acc, 4))
    print(classification_report(labels, preds, target_names=CLASSES, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()