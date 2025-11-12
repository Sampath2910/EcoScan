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

    def __init__(self, model_path='waste_classifier.pt'):
        self.class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.recyclable_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ Use /tmp directory instead of /opt — works on Render free tier
        self.cache_dir = "/tmp/model_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Full path to store the model
        self.model_path = os.path.join(self.cache_dir, model_path)

        # Google Drive backup URL for model download
        self.drive_url = "https://drive.google.com/uc?export=download&id=1JY2WJ0QdeOEUdmByj7V0Xrn4FPX5FKAz"

        # Ensure model is available
        if not os.path.exists(self.model_path):
            print("🌐 Model not found locally. Downloading from Google Drive...")
            try:
                response = requests.get(self.drive_url, stream=True)
                response.raise_for_status()
                with open(self.model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("✅ Model downloaded and saved locally.")
            except Exception as e:
                print(f"❌ Failed to download model: {e}")
                self.model_path = None

        # Load model if available
        if self.model_path and os.path.exists(self.model_path):
            self.model = self._load_model()
        else:
            self.model = None
            print("⚠️ No model available — running in mock mode.")

        # Prepare transformations
        self.transform = self._get_transforms()

    def _download_model_from_drive(self, url):
        """Downloads model weights from Google Drive (returns a model instance)."""
        print("🌐 Downloading model from Google Drive (direct)...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        model_data = BytesIO(response.content)
        # Build model architecture
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(self.class_names))

        # Try to load the file from BytesIO with explicit weights_only=False for PyTorch >=2.6
        try:
            state_or_model = torch.load(model_data, map_location=self.device, weights_only=False)
        except TypeError:
            # Older torch versions may not accept weights_only parameter
            try:
                state_or_model = torch.load(model_data, map_location=self.device)
            except Exception as e:
                raise RuntimeError(f"Failed to torch.load BytesIO: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to torch.load BytesIO: {e}")

        # If the downloaded object is a state_dict -> load_state_dict
        if isinstance(state_or_model, dict):
            model.load_state_dict(state_or_model)
        else:
            # Assume it is a full model object (less ideal) -> try to extract state_dict or load direct
            try:
                model.load_state_dict(state_or_model.state_dict())
            except Exception:
                # Attempt to use it as model directly
                model = state_or_model

        model.to(self.device)
        model.eval()
        print("✅ Model downloaded and loaded successfully (from Drive).")
        return model


    def _load_model(self):
        """Loads model locally or from cloud depending on environment."""
        try:
            # If running on Render and local cache doesn't exist, use direct download
            if os.getenv("RENDER", "false") == "true" and not os.path.exists(self.model_path):
                MODEL_URL = self.drive_url
                return self._download_model_from_drive(MODEL_URL)

            print(f"📂 Loading local model from {self.model_path}")
            # Create architecture
            model = models.resnet18(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(self.class_names))

            # Try loading with explicit weights_only=False (PyTorch >=2.6)
            try:
                state_or_model = torch.load(self.model_path, map_location=self.device, weights_only=False)
            except TypeError:
                # Older PyTorch versions: no weights_only argument
                state_or_model = torch.load(self.model_path, map_location=self.device)
            except Exception as e:
                # Re-raise with context so logs are clear
                raise RuntimeError(f"torch.load failed: {e}")

            # If we got a dict, assume it's a state_dict
            if isinstance(state_or_model, dict):
                model.load_state_dict(state_or_model)
            else:
                # If it's a full model object, try to load its state_dict; otherwise use it directly
                try:
                    model.load_state_dict(state_or_model if isinstance(state_or_model, dict) else state_or_model.state_dict())
                except Exception:
                    model = state_or_model

            model.to(self.device)
            model.eval()
            print("✅ Model loaded successfully (local).")
            return model

        except Exception as e:
            # Preserve the original exception text (useful to debug)
            print(f"❌ Failed to load model: {e}")
            # log traceback to Render logs
            import traceback as _tb
            _tb.print_exc()
            return None


    def _get_transforms(self):
        """Defines the image transformations for the ResNet model."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _get_recycling_data(self, predicted_label):
        """Determines recyclability."""
        label_lower = predicted_label.lower()
        is_rec = label_lower in self.recyclable_classes
        return {
            'label': predicted_label,
            'is_recyclable': is_rec
        }

    def predict(self, image_path, topk=1):
        """Classifies the image at the given path."""
        if self.model is None:
            return {'error': 'Model not initialized successfully.'}, []

        try:
            if not os.path.exists(image_path):
                return {'error': 'Image file not found.'}, []

            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)

            top_probs, top_indices = torch.topk(probs, topk)
            prediction_index = top_indices[0][0].item()
            prediction_label = self.class_names[prediction_index]

            result = self._get_recycling_data(prediction_label)
            raw_predictions = [{'label': self.class_names[i.item()], 'probability': p.item()}
                               for i, p in zip(top_indices[0], top_probs[0])]

            result['prediction'] = result.pop('label')
            result['confidence'] = top_probs[0][0].item() * 100
            return result, raw_predictions

        except Exception as e:
            return {'error': f'Classification error: {e}'}, []
