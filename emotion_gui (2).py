import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# -----------------------------
# Paths
# -----------------------------
BASE_PATH = r"C:\Users\singh\OneDrive\Desktop\face expresion\archive"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH  = os.path.join(BASE_PATH, "test")

# -----------------------------
# Parameters
# -----------------------------
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 150   # 🔥 Increased as requested

# -----------------------------
# Data Generators (Augmentation)
# -----------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    TEST_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# -----------------------------
# CNN Model (Stronger)
# -----------------------------
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),

    Dense(7, activation='softmax')
])

# -----------------------------
# Compile
# -----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Train
# -----------------------------
model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data
)

# -----------------------------
# Save Model
# -----------------------------
model.save("emotion_model.h5")
print("✅ Model trained for 150 epochs and saved as emotion_model.h5")
