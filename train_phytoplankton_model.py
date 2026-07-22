import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# ============ CONFIG ============
IMG_SIZE = (150, 150)
BATCH_SIZE = 16
EPOCHS = 30
LR = 0.0003
CLASSES = ["Protista", "Bacteria", "Plantae", "Not_Phytoplankton"]

# ============ PATHS ============
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "phytoplankton_dataset")  # must include Not_Phytoplankton folder
checkpoint_path = os.path.join(base_dir, "best_phytoplankton_model.h5")

# ============ DATA AUGMENTATION ============
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training', shuffle=True
)
val_gen = datagen.flow_from_directory(
    data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', shuffle=False
)

# ============ BUILD MODEL ============
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    BatchNormalization(), MaxPooling2D(),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(), MaxPooling2D(),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(), MaxPooling2D(),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer=Adam(LR), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ============ TRAIN ============
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, save_best_only=True)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)
model.save("phytoplankton_model_final_With_human.h5")

# ============ PLOT METRICS ============
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()
plt.show()
