# ================================
# 🔧 Auto-Setup + Phytoplankton Prediction
# ================================

import sys
import subprocess
import importlib

def install_if_missing(pkg):
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(f"📦 Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for pkg in ["numpy", "matplotlib", "pillow", "tqdm", "h5py", "tensorflow", "keras"]:
    install_if_missing(pkg)


# ================================
# 🔍 IMPORTS
# ================================
import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ================================
# ⚙️ CONFIG
# ================================
MODEL_PATH = "best_phytoplankton_model.h5"
IMG_SIZE = (150, 150)
CLASSES = ["Protista", "Bacteria", "Plantae"]

# ================================
# 🧠 LOAD MODEL
# ================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

print("🧠 Loading trained model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!\n")

# ================================
# 🖼️ PREDICTION FUNCTION
# ================================
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return

    print(f"📸 Predicting for: {img_path}")
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_index = np.argmax(preds[0])
    confidence = np.max(preds[0])
    predicted_class = CLASSES[class_index]

    print(f"\n🎯 Predicted Class: {predicted_class}")
    print(f"📊 Confidence: {confidence * 100:.2f}%\n")

# ================================
# 🧾 MAIN (Command-Line)
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict phytoplankton class from an image.")
    parser.add_argument("--image", required=True, help="Path to the image file")
    args = parser.parse_args()

    predict_image(args.image)
