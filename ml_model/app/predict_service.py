import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "waste_classifier.pt")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.json")
DEVICE = torch.device("cpu") # Use CPU for robust web app deployment

# --- Data Preparation (Must match validation transforms from train.py) ---
PREDICT_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Model Loading and Initialization ---
model = None
classes = []

def load_model():
    """Initializes and loads the trained ResNet-18 model and class names."""
    global model, classes
    
    # 1. Load Classes
    try:
        with open(CLASSES_PATH, "r") as f:
            classes = json.load(f)
        num_classes = len(classes)
    except FileNotFoundError:
        print(f"Error: {CLASSES_PATH} not found. Cannot load classes.")
        return False
    
    # 2. Initialize Model Structure
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 3. Load Trained Weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        print(f"PyTorch model loaded successfully with {num_classes} classes.")
        return True
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}.")
        return False
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return False

# Load the model once when the module is imported
load_model()

# --- Prediction Function ---

def predict_image(image_file_path):
    """
    Takes the path to an image file, runs the prediction, and returns the result.
    
    Args:
        image_file_path (str): The full path to the uploaded image file.
        
    Returns:
        dict: A dictionary containing 'label', 'is_recyclable', and 'suggestions'.
    """
    if model is None:
        return {'label': 'Error', 'is_recyclable': False, 'suggestions': ['Model not loaded.']}

    try:
        # Load and transform the image
        image = Image.open(image_file_path).convert("RGB")
        tensor = PREDICT_TRANSFORM(image).unsqueeze(0).to(DEVICE)

        # Run prediction
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs.data, 1)
            
        # Get the predicted class name
        class_index = predicted.item()
        predicted_label = classes[class_index]
        
        # --- Define Recyclability and Suggestions based on prediction ---
        # NOTE: This logic is simple and should be expanded for a real app.
        RECYCLABLE_MAP = {
            'plastic': {'is_recyclable': True, 'suggestions': ['Rinse and flatten bottle/container.', 'Check local collection rules.']},
            'glass': {'is_recyclable': True, 'suggestions': ['Rinse clean, remove lids.', 'Glass may require drop-off.']},
            'metal': {'is_recyclable': True, 'suggestions': ['Empty and rinse cans.', 'Crush if possible to save space.']},
            'paper': {'is_recyclable': True, 'suggestions': ['Keep dry, remove plastic windows.', 'Shredding is not usually necessary.']},
            'cardboard': {'is_recyclable': True, 'suggestions': ['Flatten boxes completely.', 'Remove all packing tape/labels.']},
            'trash': {'is_recyclable': False, 'suggestions': ['Dispose in standard waste bin.', 'No recycling options available.']},
        }
        
        result_data = RECYCLABLE_MAP.get(predicted_label.lower(), 
                                         {'is_recyclable': False, 'suggestions': ['Cannot classify or item is non-recyclable.']})
        
        result_data['label'] = predicted_label
        return result_data

    except Exception as e:
        print(f"Prediction failed: {e}")
        return {'label': 'Error', 'is_recyclable': False, 'suggestions': [f'Prediction failed: {e}']}

if __name__ == '__main__':
    # Simple test placeholder (requires a local test image)
    print("Prediction service is ready. Run me from your main app.py file.")
```
eof

### 3. Connection to the Flask App (`app.py` update needed)

You will need to import `predict_image` into your main Flask file and update your `/upload_file` route to use it.

You need to replace the mock logic in `app.py` that currently handles the file upload. The new logic will look something like this:

```python
# (In your app.py file)

# 1. Add the import
from predict_service import predict_image # Assuming predict_service.py is in the same directory

# 2. Update the upload route
@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    # ... authentication and file saving logic ...
    if request.method == 'POST':
        # ... save the uploaded file to UPLOAD_FOLDER and get file_path ...
        
        # --- NEW PREDICTION LOGIC ---
        prediction_result = predict_image(file_path)
        
        # Save prediction_result to the database (Future step, use mock storage for now)
        # current_user_history.append(prediction_result)
        
        # Redirect to the results page with the prediction data
        return redirect(url_for('show_result', 
                                label=prediction_result['label'], 
                                recyclable=prediction_result['is_recyclable'])) 

# ... rest of your Flask app code
