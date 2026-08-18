from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR = Path("data")
OUTPUT_MODEL = Path("fault_classifier.pt")
OUTPUT_CLASSES = Path("classes_names.json")

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_dataloaders():
    train_tfms= transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.486], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    return train_loader, val_loader , train_ds.classes

def build_model(num_classes: int) -> nn.Module:
    model=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(device)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total= 0.0, 0, 0

    for images,labels in loader:
        images,labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss= criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()* images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

        return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
 
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
 
    return running_loss / total, correct / total

def main():
    train_loader, val_loader, class_names = build_dataloaders()
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(num_classes=len(class_names))
    criterion=nn.CrossEntropyLoss()

    optimizer= torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)

        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )  

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_MODEL)
            OUTPUT_CLASSES.write_text(json.dumps(class_names, indent=2))
            print(f"  -> new best model saved (val_acc={val_acc:.3f})")
 
    print(f"Training complete. Best val_acc={best_val_acc:.3f}")
    print(f"Saved: {OUTPUT_MODEL}, {OUTPUT_CLASSES}")
 
 
if __name__ == "__main__":
    main()