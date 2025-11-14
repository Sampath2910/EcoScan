import requests, torch
from io import BytesIO

drive_url = "https://drive.google.com/file/d/1BmNNnZQLqGP_lkSBGieAE7pyhRUCuZXn/view?usp=sharing"

print("🌐 Downloading model from Google Drive...")
response = requests.get(drive_url, stream=True, timeout=60)
response.raise_for_status()

# Try to load model weights directly
model_bytes = BytesIO(response.content)
state_dict = torch.load(model_bytes, map_location="cpu", weights_only=False)

if isinstance(state_dict, dict):
    print(f"✅ Model loaded successfully! Keys found: {len(state_dict.keys())}")
else:
    print("⚠️ Warning: File did not return a state_dict — might be a full model object.")

print("🎉 Download and load test complete.")
