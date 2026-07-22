import os
import numpy as np
import tensorflow as tf
import shutil
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============ CONFIG ============
IMG_SIZE = (150, 150)
BATCH_SIZE = 8  # Lowered for memory efficiency
EPOCHS = 10     # Start small; raise once it's stable
LR = 0.001
CLASSES = ["Protista", "Bacteria", "Plantae"]

# ============ PATHS ============
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "phytoplankton_dataset")
output_dir = os.path.join(base_dir, "classified_kingdoms")
os.makedirs(output_dir, exist_ok=True)
for label in CLASSES:
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)

# ============ DATA ============
datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ============ MODEL ============
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(len(CLASSES), activation='softmax')
])
model.compile(optimizer=Adam(LR), loss='categorical_crossentropy', metrics=['accuracy'])
# ============ SAVE MODEL ============
model.save("phytoplankton_classifier_model.h5")
print("\n💾 Model saved as phytoplankton_classifier_model.h5")
# ============ TRAIN ============
print("\n📚 Training the CNN model...\n")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[EarlyStopping(patience=4, restore_best_weights=True)]
)

# ============ CLASSIFY ============
index_to_class = {v: k for k, v in train_gen.class_indices.items()}
print("\n🔍 Classifying images and copying to kingdom folders...\n")
count = {k: 0 for k in CLASSES}

for root, _, files in os.walk(dataset_dir):
    for file in tqdm(files, desc="Processing images"):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(root, file)
            try:
                img = tf.keras.preprocessing.image.load_img(full_path, target_size=IMG_SIZE)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                pred = model.predict(img_array, verbose=0)
                class_index = np.argmax(pred[0])
                confidence = np.max(pred[0])
                label = index_to_class[class_index]
                dest = os.path.join(output_dir, label, file)
                shutil.copy2(full_path, dest)
                count[label] += 1
                print(f"✅ {file} → {label} ({confidence:.2f} confidence)")
            except Exception as e:
                print(f"❌ Error with {file}: {e}")

# ============ SUMMARY ============
print("\n📊 Classification Summary:")
for k, v in count.items():
    print(f"  {k}: {v} images")
print(f"\n🎉 Done! Check the folder: {output_dir}")
