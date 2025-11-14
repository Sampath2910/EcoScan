import torch
from torchvision import models
import torch.nn as nn

state_dict = torch.load("waste_classifier.pth", map_location="cpu")

model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)

model.load_state_dict(state_dict)
model.eval()

# Dynamic quantization (make it lightweight)
model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

example = torch.randn(1, 3, 224, 224)

traced = torch.jit.trace(model, example)
traced.save("waste_classifier_ts.pt")

print("Saved waste_classifier_ts.pt")
