# evaluate.py
import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import json

# optional: classification_report
try:
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

BASE_DIR = os.path.dirname(__file__)
dataset_path = os.path.join(BASE_DIR, "dataset-resized")
model_path = os.path.join(BASE_DIR, "waste_classifier.pt")
classes_path = os.path.join(BASE_DIR, "classes.json")
split_path = os.path.join(BASE_DIR, "split_indices.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# transforms for val/test
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# load class names
if os.path.exists(classes_path):
    with open(classes_path, "r") as f:
        classes = json.load(f)
else:
    # fallback
    dataset_tmp = datasets.ImageFolder(root=dataset_path)
    classes = dataset_tmp.classes

# load split indices
if os.path.exists(split_path):
    splits = torch.load(split_path)
    test_indices = splits.get("test", None)
else:
    test_indices = None

dataset = datasets.ImageFolder(root=dataset_path, transform=val_transform)

if test_indices is not None:
    test_subset = Subset(dataset, test_indices)
else:
    # fallback: recreate random split (less ideal)
    from torch.utils.data import random_split
    total = len(dataset)
    t = int(0.7 * total)
    v = int(0.15 * total)
    _, _, test_subset = random_split(dataset, [t, v, total - t - v])

test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
print(f"Classes: {classes}")
print(f"Test samples: {len(test_subset)}")

# load model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
num_classes = len(classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# evaluate
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.tolist())

total = len(all_labels)
correct = sum([1 for p, t in zip(all_preds, all_labels) if p == t])
accuracy = 100.0 * correct / max(1, total)
print(f"✅ Test Accuracy: {accuracy:.2f}%")

# optional detailed report
if SKLEARN_AVAILABLE:
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))
    print("\nConfusion matrix:")
    print(confusion_matrix(all_labels, all_preds))
else:
    print("sklearn not available — install scikit-learn to get classification_report & confusion_matrix.")
