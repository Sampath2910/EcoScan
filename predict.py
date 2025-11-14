# predict.py

import torch
from torchvision import models, transforms
from PIL import Image
import os
import torch.nn as nn
import requests

class WasteClassifier:
    """
    Optimized Waste Classifier with:
    - Lazy model loading (fixes Render timeout)
    - TorchScript support (faster, lower memory)
    - Automatic download fallback for .pth
    """

    def __init__(self, model_path='waste_classifier_ts.pt'):
        self.class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.recyclable_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cache directory on Render
        self.cache_dir = "/tmp/model_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Paths for TorchScript + fallback .pth
        self.model_path = os.path.join(self.cache_dir, model_path)
        self.fallback_pth_path = os.path.join(self.cache_dir, "waste_classifier.pth")

        # HuggingFace model (.pth)
        self.model_url = "https://huggingface.co/Sampath2910/EcoScan-Classifier/resolve/main/waste_classifier.pth?download=true"

        # Start with model unloaded → Lazy loading
        self.model = None  

        # Preload transforms immediately
        self.transform = self._get_transforms()

        print("🔄 WasteClassifier initialized (lazy loading mode).")

    # -------------------------------------------------------------------
    # DOWNLOAD MODEL FILE (.pth)
    # -------------------------------------------------------------------
    def _download_model_file(self):
        try:
            print("🌐 Downloading model (.pth) from HuggingFace...")
            response = requests.get(self.model_url, stream=True, timeout=90)
            response.raise_for_status()

            with open(self.fallback_pth_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print("✅ Downloaded .pth model:", self.fallback_pth_path)
            return True
        except Exception as e:
            print("❌ Model download failed:", e)
            return False

    # -------------------------------------------------------------------
    # LOAD TORCHSCRIPT MODEL IF AVAILABLE (FAST)
    # -------------------------------------------------------------------
    def _load_torchscript(self):
        try:
            if os.path.exists(self.model_path):
                print("📂 Loading TorchScript model from:", self.model_path)
                model = torch.jit.load(self.model_path, map_location=self.device)
                model.eval()
                print("✅ TorchScript model loaded successfully!")
                return model
            return None
        except Exception as e:
            print("⚠️ TorchScript load failed:", e)
            return None

    # -------------------------------------------------------------------
    # LOAD FALLBACK .PTH → Convert → Load
    # -------------------------------------------------------------------
    def _load_and_convert_pth(self):
        try:
            if not os.path.exists(self.fallback_pth_path):
                # Download if not present
                if not self._download_model_file():
                    return None

            print("📦 Loading .pth model for conversion:", self.fallback_pth_path)

            # Load ResNet18
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(self.class_names))

            state_dict = torch.load(self.fallback_pth_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()

            # Quantize to reduce size & RAM
            model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

            # Convert to TorchScript
            example = torch.randn(1, 3, 224, 224)
            traced = torch.jit.trace(model, example)

            traced.save(self.model_path)
            print("✅ Converted and saved TorchScript model:", self.model_path)

            return traced

        except Exception as e:
            print("❌ Failed loading/converting .pth:", e)
            return None

    # -------------------------------------------------------------------
    # LAZY LOAD MODEL (only at first prediction)
    # -------------------------------------------------------------------
    def _ensure_model_loaded(self):
        if self.model is not None:
            return  # Already loaded

        print("⏳ Lazy-loading model on demand...")

        # 1. Try loading fast TorchScript version
        self.model = self._load_torchscript()
        if self.model:
            return

        # 2. If not present → load/convert .pth
        self.model = self._load_and_convert_pth()
        if self.model:
            return

        print("❌ FATAL: No model could be loaded.")
        self.model = None

    # -------------------------------------------------------------------
    # TRANSFORMS
    # -------------------------------------------------------------------
    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    # -------------------------------------------------------------------
    # CLASSIFY IMAGE
    # -------------------------------------------------------------------
    def predict(self, image_path, topk=1):

        # Lazy model load (critical)
        self._ensure_model_loaded()

        # If still not loaded → use mock fallback
        if self.model is None:
            print("⚠️ Model unavailable → Returning mock result.")
            return {
                "error": "Model unavailable (fallback).",
                "prediction": "trash",
                "confidence": 0.0,
                "label": "trash",
                "is_recyclable": False,
            }, []

        try:
            if not os.path.exists(image_path):
                return {"error": "Image not found"}, []

            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)

            top_probs, top_indices = torch.topk(probs, topk)
            idx = top_indices[0][0].item()
            label = self.class_names[idx]

            result = {
                "prediction": label,
                "confidence": float(top_probs[0][0].item() * 100),
                "label": label,
                "is_recyclable": label.lower() in self.recyclable_classes
            }

            details = [
                {
                    "label": self.class_names[i.item()],
                    "probability": float(p.item())
                }
                for i, p in zip(top_indices[0], top_probs[0])
            ]

            return result, details

        except Exception as e:
            print("❌ Classification error:", e)
            return {"error": str(e)}, []
