# train.py
import os
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

logging.basicConfig(level=logging.INFO)

# reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 1. Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 2. Dataset
dataset_path = os.path.join(os.path.dirname(__file__), "dataset-resized")
logging.info(f"Dataset path: {dataset_path}")
full_dataset = datasets.ImageFolder(root=dataset_path)
classes = full_dataset.classes
num_classes = len(classes)
logging.info(f"Classes ({num_classes}): {classes}")

# 3. Create deterministic splits (70/15/15) and save indices
num_samples = len(full_dataset)
indices = torch.randperm(num_samples).tolist()
train_size = int(0.7 * num_samples)
val_size = int(0.15 * num_samples)
test_size = num_samples - train_size - val_size

train_idx = indices[:train_size]
val_idx = indices[train_size:train_size + val_size]
test_idx = indices[train_size + val_size:]

train_dataset = Subset(datasets.ImageFolder(root=dataset_path, transform=train_transform), train_idx)
val_dataset   = Subset(datasets.ImageFolder(root=dataset_path, transform=val_transform),   val_idx)
test_dataset  = Subset(datasets.ImageFolder(root=dataset_path, transform=val_transform),   test_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

logging.info(f"Split sizes -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# 4. Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
for param in model.parameters():
    param.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# 5. Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# 6. Training with early stopping
EPOCHS = 5
best_val_acc = 0.0
patience = 3
no_improve = 0
BEST_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / max(1, len(train_loader))

    # validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100.0 * correct / max(1, total)
    logging.info(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.2f}%")

    # early stopping logic
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve = 0
        torch.save(model.state_dict(), BEST_PATH)
    else:
        no_improve += 1

    if no_improve >= patience:
        logging.info("Early stopping triggered.")
        break

    scheduler.step()

# Load best model (if exists) and save final artifacts
if os.path.exists(BEST_PATH):
    model.load_state_dict(torch.load(BEST_PATH, map_location=device))

SAVE_PATH = os.path.join(os.path.dirname(__file__), "waste_classifier.pt")
torch.save(model.state_dict(), SAVE_PATH)
logging.info(f"Model saved: {SAVE_PATH}")

# save classes and split indices for reproducibility
with open(os.path.join(os.path.dirname(__file__), "classes.json"), "w") as f:
    json.dump(classes, f)
torch.save({"train": train_idx, "val": val_idx, "test": test_idx},
           os.path.join(os.path.dirname(__file__), "split_indices.pth"))

logging.info("Classes and split indices saved.")
