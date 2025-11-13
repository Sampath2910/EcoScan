# predict.py

import torch
from torchvision import models, transforms
from PIL import Image
import os
import torch.nn as nn
import requests
from io import BytesIO


class WasteClassifier:
    """
    Initializes and manages the PyTorch ResNet18 model for waste classification.
    """

    def __init__(self, model_path='waste_classifier.pth'):
        self.class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.recyclable_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cache directory works on Render free tier
        self.cache_dir = "/tmp/model_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Local path to store the model
        self.model_path = os.path.join(self.cache_dir, model_path)

        # HuggingFace model file URL (direct download)
        self.model_url = "https://huggingface.co/Sampath2910/EcoScan-Classifier/resolve/main/waste_classifier.pth"

        # Ensure the model is present locally
        if not os.path.exists(self.model_path):
            print("🌐 Model not found locally. Downloading from HuggingFace...")
            if not self._download_model_file():
                print("❌ Model download failed — using mock mode.")
                self.model = None
                return

        # Load the model
        self.model = self._load_model()

        if self.model is None:
            print("⚠️ Model failed to load — running in mock mode.")
        else:
            print("✅ WasteClassifier initialized successfully.")

        # Image transforms
        self.transform = self._get_transforms()

    # -----------------------------------------------------
    # DOWNLOAD THE MODEL FILE
    # -----------------------------------------------------
    def _download_model_file(self):
        try:
            response = requests.get(self.model_url, stream=True, timeout=60)
            response.raise_for_status()

            with open(self.model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print("✅ Model downloaded and saved at:", self.model_path)
            return True

        except Exception as e:
            print("❌ Failed to download model from HuggingFace:", e)
            return False

    # -----------------------------------------------------
    # LOAD THE DOWNLOADED MODEL SAFELY
    # -----------------------------------------------------
    def _load_model(self):
        try:
            print(f"📂 Loading local model from {self.model_path}")

            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(self.class_names))

            # PyTorch 2.6 fix — allow loading full pickle
            try:
                state_dict = torch.load(
                    self.model_path,
                    map_location=self.device,
                    weights_only=False
                )
            except TypeError:
                state_dict = torch.load(self.model_path, map_location=self.device)

            if not isinstance(state_dict, dict):
                print("⚠️ Model file is not a state_dict — invalid format.")
                return None

            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            print("✅ Model loaded successfully.")
            return model

        except Exception as e:
            print("❌ Failed to load model:", e)
            return None

    # -----------------------------------------------------
    # IMAGE TRANSFORMS
    # -----------------------------------------------------
    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    # -----------------------------------------------------
    # DETERMINE RECYCLABILITY
    # -----------------------------------------------------
    def _get_recycling_data(self, predicted_label):
        is_rec = predicted_label.lower() in self.recyclable_classes
        return {"label": predicted_label, "is_recyclable": is_rec}

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------
    def predict(self, image_path, topk=1):
        if self.model is None:
            return {"error": "Model not initialized successfully."}, []

        try:
            if not os.path.exists(image_path):
                return {"error": "Image file not found."}, []

            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)

            top_probs, top_indices = torch.topk(probs, topk)
            idx = top_indices[0][0].item()

            label = self.class_names[idx]

            result = self._get_recycling_data(label)
            result["prediction"] = label
            result["confidence"] = float(top_probs[0][0].item() * 100)

            details = [
                {
                    "label": self.class_names[i.item()],
                    "probability": float(p.item())
                }
                for i, p in zip(top_indices[0], top_probs[0])
            ]

            return result, details

        except Exception as e:
            return {"error": f"Classification error: {e}"}, []
