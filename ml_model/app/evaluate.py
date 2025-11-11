# evaluate.py
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import resnet18, ResNet18_Weights
import os

# -------------------------
# 1. Paths
# -------------------------
folder = os.path.dirname(__file__)
dataset_path = os.path.join(folder, "dataset-resized")   # same dataset as training
model_path = os.path.join(folder, "waste_classifier.pt")

# -------------------------
# 2. Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------
# 3. Transformations
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# 4. Dataset & Splits
# -------------------------
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
classes = dataset.classes

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

_, _, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

print(f"Classes: {classes}")
print(f"Test samples: {len(test_data)}")

# -------------------------
# 5. Load Model
# -------------------------
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
num_classes = len(classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# -------------------------
# 6. Evaluate
# -------------------------
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")
