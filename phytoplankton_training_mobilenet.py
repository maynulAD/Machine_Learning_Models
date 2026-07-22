# ================================
# 🌿 Phytoplankton Classifier (MobileNetV2)
# ================================
# Works on CPU / Anaconda Spyder
# Dataset structure:
# phytoplankton_dataset/
#     Protista/
#     Bacteria/
#     Plantae/
# Each folder contains its class images

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight

# ================================
# ⚙️ CONFIGURATION
# ================================
IMG_SIZE = (150, 150)
BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 1e-4
CLASSES = ["Protista", "Bacteria", "Plantae"]
DATASET_DIR = "phytoplankton_dataset"
MODEL_SAVE_PATH = "best_phytoplankton_model.h5"

# ================================
# 🧩 DATA PREPARATION
# ================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    subset='validation',
    shuffle=False
)

# ================================
# ⚖️ CLASS WEIGHT (Handle Imbalance)
# ================================
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))
print("\n📊 Class Weights:", class_weights)

# ================================
# 🧠 MODEL (MobileNetV2 Transfer Learning)
# ================================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150,150,3))
base_model.trainable = False  # Freeze convolutional base

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ================================
# ⏳ TRAINING
# ================================
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
]

print("\n🚀 Training started...\n")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

print("\n✅ Training complete!")
print(f"Best model saved at: {MODEL_SAVE_PATH}")

# ================================
# 🔍 EVALUATE MODEL
# ================================
loss, acc = model.evaluate(val_gen)
print(f"\n📈 Validation Accuracy: {acc*100:.2f}%")
print(f"📉 Validation Loss: {loss:.4f}")

# ================================
# 🧠 (Optional) Fine-tune top layers
# ================================
print("\n🔧 Fine-tuning the top layers of MobileNetV2...\n")

base_model.trainable = True
for layer in base_model.layers[:-40]:  # unfreeze only top 40 layers
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)

loss, acc = model.evaluate(val_gen)
print(f"\n🎯 Final Validation Accuracy after fine-tuning: {acc*100:.2f}%")
print(f"💾 Final model saved at: {MODEL_SAVE_PATH}")
