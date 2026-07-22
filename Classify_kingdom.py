import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration
CONFIG = {
    "image_size": (150, 150),
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 0.001,
    "validation_split": 0.2,
    "class_names": ["Protista", "Bacteria", "Plantae"]
}

# Path setup
def setup_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        "dataset": os.path.join(base_dir, "phytoplankton_dataset"),
        "output": os.path.join(base_dir, "classified_kingdoms"),
        "model": os.path.join(base_dir, "phytoplankton_classifier.h5")
    }

# Data preparation
def prepare_data(paths):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=CONFIG["validation_split"],
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        paths["dataset"],
        target_size=CONFIG["image_size"],
        batch_size=CONFIG["batch_size"],
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        paths["dataset"],
        target_size=CONFIG["image_size"],
        batch_size=CONFIG["batch_size"],
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    return train_generator, val_generator

# CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(*CONFIG["image_size"], 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(len(CONFIG["class_names"]), activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=CONFIG["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# Training
def train_model(model, train_gen, val_gen, paths):
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(paths["model"], save_best_only=True)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG["epochs"],
        callbacks=callbacks
    )

    # Plot training history
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.legend()

    plt.savefig("training_history.png")
    plt.close()

# Classification and organization
def classify_and_organize(model, paths):
    # Create output directories
    for class_name in CONFIG["class_names"]:
        os.makedirs(os.path.join(paths["output"], class_name), exist_ok=True)

    # Get class mapping
    class_indices = {v:k for k,v in train_generator.class_indices.items()}

    # Process images
    for root, _, files in os.walk(paths["dataset"]):import os
import sys
import subprocess
import numpy as np
import shutil
from tqdm import tqdm

# Check and install missing packages
required_packages = {
    'tensorflow': 'tensorflow',
    'matplotlib': 'matplotlib',
    'numpy': 'numpy',
    'tqdm': 'tqdm'
}

missing_packages = []
for package, import_name in required_packages.items():
    try:
        __import__(import_name)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f"Installing missing packages: {', '.join(missing_packages)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])

# Now import the rest
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Configuration (same as before)
CONFIG = {
    "image_size": (150, 150),
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 0.001,
    "validation_split": 0.2,
    "class_names": ["Protista", "Bacteria", "Plantae"]
}

def setup_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        "dataset": os.path.join(base_dir, "phytoplankton_dataset"),
        "output": os.path.join(base_dir, "classified_kingdoms"),
        "model": os.path.join(base_dir, "phytoplankton_classifier.h5")
    }

def prepare_data(paths):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=CONFIG["validation_split"],
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        paths["dataset"],
        target_size=CONFIG["image_size"],
        batch_size=CONFIG["batch_size"],
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        paths["dataset"],
        target_size=CONFIG["image_size"],
        batch_size=CONFIG["batch_size"],
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    return train_generator, val_generator

def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(*CONFIG["image_size"], 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(len(CONFIG["class_names"]), activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=CONFIG["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def train_model(model, train_gen, val_gen, paths):
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(paths["model"], save_best_only=True)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG["epochs"],
        callbacks=callbacks
    )
    
    # Only plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Val Accuracy")
        plt.title("Accuracy")
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.title("Loss")
        plt.legend()

        plt.savefig("training_history.png")
        plt.close()
    except ImportError:
        print("Matplotlib not available - skipping training plots")

def classify_and_organize(model, paths):
    for class_name in CONFIG["class_names"]:
        os.makedirs(os.path.join(paths["output"], class_name), exist_ok=True)

    class_indices = {v:k for k,v in train_generator.class_indices.items()}

    for root, _, files in os.walk(paths["dataset"]):
        for file in tqdm(files, desc="Classifying images"):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    img_path = os.path.join(root, file)
                    img = tf.keras.preprocessing.image.load_img(
                        img_path, target_size=CONFIG["image_size"])
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0

                    pred = model.predict(img_array)
                    pred_class = np.argmax(pred[0])
                    class_name = class_indices[pred_class]

                    dest_path = os.path.join(paths["output"], class_name, file)
                    shutil.copy2(img_path, dest_path)

                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    paths = setup_paths()
    train_generator, val_generator = prepare_data(paths)
    model = build_model()
    train_model(model, train_generator, val_generator, paths)
    classify_and_organize(model, paths)
    print(f"\nClassification complete! Images organized in: {paths['output']}")
        for file in tqdm(files, desc="Classifying images"):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    img_path = os.path.join(root, file)
                    img = tf.keras.preprocessing.image.load_img(
                        img_path, target_size=CONFIG["image_size"])
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0

                    pred = model.predict(img_array)
                    pred_class = np.argmax(pred[0])
                    class_name = class_indices[pred_class]

                    dest_path = os.path.join(paths["output"], class_name, file)
                    shutil.copy2(img_path, dest_path)

                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

# Main execution
if __name__ == "__main__":
    # Setup paths
    paths = setup_paths()

    # Prepare data
    train_generator, val_generator = prepare_data(paths)

    # Build and train model
    model = build_model()
    train_model(model, train_generator, val_generator, paths)

    # Classify and organize images
    classify_and_organize(model, paths)

    print(f"\nClassification complete! Images organized in: {paths['output']}")