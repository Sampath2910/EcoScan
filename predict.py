# predict.py

import torch
from torchvision import models, transforms
from PIL import Image
import os
import torch.nn as nn # Need to import nn for customizing the final layer

class WasteClassifier:
    """
    Initializes and manages the PyTorch ResNet18 model for waste classification.
    """
    def __init__(self, model_path='waste_classifier.pt'):
        # Define class_names and recyclable_classes directly within the class
        self.class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'] 
        self.recyclable_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        self.model = self._load_model()
        self.transform = self._get_transforms()
        

    def _load_model(self):
        """Loads the ResNet18 model architecture and weights."""
        try:
            # Load the pre-trained model architecture (using Weights.DEFAULT is modern, but using pretrained=False 
            # and then loading custom state_dict is correct for trained models)
            model = models.resnet18(pretrained=False)
            
            # Adjust the final layer to match the number of classes
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(self.class_names)) 
            
            # Load the state dictionary from the .pt file
            # Map location ensures the file loads correctly regardless of CUDA availability
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval() # Set the model to evaluation mode
            return model
        except FileNotFoundError:
            # Re-raise FileNotFoundError, which is handled in app.py's global CLASSIFIER block
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        except Exception as e:
            raise Exception(f"Failed to load model weights: {e}")

    def _get_transforms(self):
        """Defines the necessary image transformations for the ResNet model."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # Standard normalization for ImageNet pre-trained models
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def _get_recycling_data(self, predicted_label):
        """Determines recyclability based on the predicted label."""
        label_lower = predicted_label.lower()
        
        # NOTE: Suggestions are now generated in app.py's classify_image for separation of concerns, 
        # but we use this map to confirm recyclability.
        RECYCLABLE_CLASSES = self.recyclable_classes
        
        is_rec = label_lower in RECYCLABLE_CLASSES
        
        return {
            'label': predicted_label, 
            'is_recyclable': is_rec
        }

    def predict(self, image_path, topk=1):
        """
        Classifies the image at the given path.
        Returns: Tuple of (classification_result: dict, raw_predictions: list)
        """
        if self.model is None:
            return {'error': 'Model not initialized successfully.'}, []
            
        try:
            if not os.path.exists(image_path):
                return {'error': 'Image file not found.'}, []

            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                # Convert logits to probabilities
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
            # Get top K predictions
            top_probs, top_indices = torch.topk(probs, topk)
            
            prediction_index = top_indices[0][0].item()
            prediction_label = self.class_names[prediction_index]

            # Get base result data
            result = self._get_recycling_data(prediction_label)
            
            # Format raw predictions for detailed feedback if needed
            raw_predictions = [{'label': self.class_names[i.item()], 'probability': p.item()} 
                                 for i, p in zip(top_indices[0], top_probs[0])]

            # Rename 'label' key to 'prediction' to match the original utility function signature in app.py
            result['prediction'] = result.pop('label')
            result['confidence'] = top_probs[0][0].item() * 100 
            
            return result, raw_predictions

        except Exception as e:
            # Handle potential image loading or processing errors
            return {'error': f'Classification error: {e}'}, []
