import torch
from torchvision import models
import torch.nn as nn

MODEL_IN = "waste_classifier.pth"
MODEL_OUT = "waste_classifier_optimized.pth"
SCRIPTED_OUT = "waste_classifier_scripted.pt"

print("Loading original model...")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)

state_dict = torch.load(MODEL_IN, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

print("Applying dynamic quantization...")
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

print("Saving quantized model...")
torch.save(quantized_model.state_dict(), MODEL_OUT)

print("Converting to TorchScript...")
scripted_model = torch.jit.script(quantized_model)
scripted_model.save(SCRIPTED_OUT)

print("DONE!")
print("Generated:")
print(" -", MODEL_OUT)
print(" -", SCRIPTED_OUT)
