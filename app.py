import io
import os
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_file
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = Flask(__name__)

# Constants
N_CLASSES = 11
MODEL_PATH = 'best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading model from {MODEL_PATH} onto {DEVICE}...")
            model = smp.DeepLabV3Plus(
                encoder_name="mit_b2",
                encoder_weights=None,
                in_channels=3,
                classes=N_CLASSES,
            ).to(DEVICE)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return False

# Initialize model
load_model_success = load_model()
if not load_model_success:
    print(f"WARNING: Model weights '{MODEL_PATH}' not found. Inference will return mock data.")

# Validation transforms from training script
def get_val_augmentation(h=512, w=512):
    return A.Compose([
        A.Resize(h, w, interpolation=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# Map class IDs to predefined colors for the frontend
CLASS_COLORS = {
    0:  [0, 0, 0],         # Background (Black)
    1:  [34, 139, 34],     # Trees (Forest Green)
    2:  [50, 205, 50],     # Lush Bushes (Lime Green)
    3:  [218, 165, 32],    # Dry Grass (Goldenrod)
    4:  [184, 134, 11],    # Dry Bushes (Dark Goldenrod)
    5:  [160, 82, 45],     # Ground Clutter (Sienna)
    6:  [255, 20, 147],    # Flowers (Deep Pink)
    7:  [139, 69, 19],     # Logs (Saddle Brown)
    8:  [128, 128, 128],   # Rocks (Gray)
    9:  [210, 180, 140],   # Landscape (Tan)
    10: [135, 206, 235],   # Sky (Sky Blue)
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Load image
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        orig_w, orig_h = pil_img.size
        
        img_array = np.array(pil_img)

        # Preprocess
        transform = get_val_augmentation(512, 512)
        augmented = transform(image=img_array)
        tensor_img = augmented['image'].unsqueeze(0).to(DEVICE)

        if model is not None:
            # Run inference
            model.eval()
            with torch.no_grad():
                outputs = model(tensor_img)
                # outputs shape: (1, N_CLASSES, 512, 512)
                
                # Resize back to original image size
                import torch.nn.functional as F
                outputs = F.interpolate(outputs, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
                
                # Get class predictions
                preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy() # (H, W)
                
        else:
            return jsonify({'error': f"Model weights '{MODEL_PATH}' not found. Please download from Kaggle."}), 500

        # Convert class indices to RGBA Image (Make Background Transparent)
        color_mask = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
        for class_idx, color in CLASS_COLORS.items():
            if class_idx == 0:  # Background class
                color_mask[preds == class_idx] = [0, 0, 0, 0] # Transparent
            else:
                color_mask[preds == class_idx] = color + [180] # Add some transparency (alpha=180)

        # Save to memory and return
        mask_img = Image.fromarray(color_mask, 'RGBA')
        img_io = io.BytesIO()
        mask_img.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
