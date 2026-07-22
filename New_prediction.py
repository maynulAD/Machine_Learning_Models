import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# ============ CONFIG ============
IMG_SIZE = (150, 150)
CLASSES = ["Protista", "Bacteria", "Plantae"]
PHYTOTHRESHOLD = 0.60  # Minimum confidence to trust it's a phytoplankton
model_path = "cnn_phytoplankton_final.h5"

# ============ LOAD MODEL ============
model = load_model(model_path)
print("✅ Model loaded successfully!")

# ============ INPUT FROM CMD ============
if len(sys.argv) < 2:
    print("❗ Please provide image path as argument.")
    print("👉 Example: python predict_kingdom.py path/to/image.png")
    sys.exit()

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print(f"❌ Image file not found: {image_path}")
    sys.exit()

try:
    # Load and preprocess image
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array, verbose=0)[0]
    class_index = np.argmax(prediction)
    raw_confidence = float(np.max(prediction))
    label = CLASSES[class_index]

    # ============ DECISION LOGIC ============
    if raw_confidence >= PHYTOTHRESHOLD:
        print(f"\n🧬 Prediction: {label}")
        print(f"✅ Confidence: {raw_confidence * 100:.2f}%")
    else:
        print(f"\n❌ Not a phytoplankton (Confidence: {raw_confidence * 100:.2f}%)")
        print(f"ℹ️ Most likely class (not trusted): {label} ({raw_confidence * 100:.2f}%)")

except Exception as e:
    print(f"\n🚫 Error processing image:\n{e}")
