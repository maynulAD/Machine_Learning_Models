import sys, os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE = (150, 150)
CLASSES = ["Protista", "Bacteria", "Plantae", "Not_Phytoplankton"]
MODEL_PATH = "phytoplankton_model_final_With_human.h5"
CONF_THRESHOLD = 0.60

def main(img_path):
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        return

    model = load_model(MODEL_PATH)
    img = load_img(img_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr, verbose=0)[0]
    idx = np.argmax(pred)
    conf = float(pred[idx])
    label = CLASSES[idx]

    if label == "Not_Phytoplankton" or conf < CONF_THRESHOLD:
        print(f"\n❌ Result: Not a phytoplankton")
        print(f"🔍 Most likely: {label} ({conf*100:.2f}%)")
    else:
        print(f"\n✅ Result: {label}")
        print(f"📈 Confidence: {conf*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_image.py path/to/image.jpg")
        sys.exit()
    main(sys.argv[1])

