# predict.py

import torch
from torchvision import models, transforms
from PIL import Image
import os
import torch.nn as nn
import requests


class WasteClassifier:
    """
    Optimized Waste Classifier:
    - Lazy model loading (fixes Render timeout)
    - TorchScript support (fast, small, CPU friendly)
    - Safe fallback .pth load & convert
    - NO downloads during predict()
    """

    def __init__(self, model_path='waste_classifier_scripted.pt'):
        self.class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.recyclable_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

        # Render = CPU only
        self.device = torch.device("cpu")

        # Cache folder
        self.cache_dir = "/tmp/model_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Local model paths
        self.model_path = os.path.join(self.cache_dir, model_path)
        self.fallback_pth_path = os.path.join(self.cache_dir, "waste_classifier_optimized.pth")

        # -----------------------
        # 🔥 USE YOUR NEW URLs
        # -----------------------
        self.ts_url = (
            "https://huggingface.co/Sampath2910/EcoScan-Classifier/resolve/main/waste_classifier_scripted.pt?download=true"
        )

        self.pth_url = (
            "https://huggingface.co/Sampath2910/EcoScan-Classifier/resolve/main/waste_classifier_optimized.pth?download=true"
        )

        # Lazy-loaded model
        self.model = None

        # Load transforms immediately
        self.transform = self._get_transforms()

        print("🔄 WasteClassifier initialized (lazy loading mode).")

    # -----------------------------------------------------------
    # DOWNLOAD FILE SAFELY (ONLY IF MISSING)
    # -----------------------------------------------------------
    def _download_if_missing(self, url, path):
        if os.path.exists(path):
            return True

        try:
            print(f"🌐 Downloading model from: {url}")
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()

            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            print("✅ Downloaded:", path)
            return True

        except Exception as e:
            print("❌ Download failed:", e)
            return False

    # -----------------------------------------------------------
    # LOAD TORCHSCRIPT (FASTEST)
    # -----------------------------------------------------------
    def _load_torchscript(self):
        try:
            self._download_if_missing(self.ts_url, self.model_path)

            if not os.path.exists(self.model_path):
                return None

            print("📂 Loading TorchScript:", self.model_path)
            model = torch.jit.load(self.model_path, map_location=self.device)
            model.eval()
            print("✅ TorchScript model loaded.")
            return model

        except Exception as e:
            print("⚠️ TorchScript load failed:", e)
            return None

    # -----------------------------------------------------------
    # LOAD .PTH → QUANTIZE → TORCHSCRIPT CONVERT
    # -----------------------------------------------------------
    def _load_and_convert_pth(self):
        try:
            if not os.path.exists(self.fallback_pth_path):
                if not self._download_if_missing(self.pth_url, self.fallback_pth_path):
                    return None

            print("📦 Loading .pth model:", self.fallback_pth_path)

            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(self.class_names))

            state_dict = torch.load(self.fallback_pth_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()

            # Quantization step
            model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

            # Convert → TorchScript
            example = torch.randn(1, 3, 224, 224)
            traced = torch.jit.trace(model, example)

            traced.save(self.model_path)
            print("✅ Converted → TorchScript saved:", self.model_path)

            return traced

        except Exception as e:
            print("❌ .pth conversion failed:", e)
            return None

    # -----------------------------------------------------------
    # LAZY LOAD (first prediction only)
    # -----------------------------------------------------------
    def _ensure_model_loaded(self):
        if self.model is not None:
            return

        print("⏳ Lazy-loading model...")

        # Try TorchScript first
        self.model = self._load_torchscript()
        if self.model:
            return

        # Otherwise try .pth
        self.model = self._load_and_convert_pth()
        if self.model:
            return

        print("❌ No model could be loaded.")
        self.model = None

    # -----------------------------------------------------------
    # TRANSFORMS
    # -----------------------------------------------------------
    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    # -----------------------------------------------------------
    # PREDICT
    # -----------------------------------------------------------
    def predict(self, image_path, topk=1):

        self._ensure_model_loaded()

        if self.model is None:
            print("⚠️ Model unavailable → mock fallback.")
            return {
                "error": "Model unavailable",
                "prediction": "trash",
                "label": "trash",
                "confidence": 0.0,
                "is_recyclable": False
            }, []

        try:
            if not os.path.exists(image_path):
                return {"error": "Image not found"}, []

            img = Image.open(image_path).convert("RGB")
            tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)

            top_probs, top_idx = torch.topk(probs, topk)
            idx = top_idx[0][0].item()
            label = self.class_names[idx]

            result = {
                "prediction": label,
                "label": label,
                "confidence": float(top_probs[0][0].item() * 100),
                "is_recyclable": label.lower() in self.recyclable_classes
            }

            details = [
                {
                    "label": self.class_names[i.item()],
                    "probability": float(p.item())
                }
                for i, p in zip(top_idx[0], top_probs[0])
            ]

            return result, details

        except Exception as e:
            print("❌ Prediction error:", e)
            return {"error": str(e)}, []
