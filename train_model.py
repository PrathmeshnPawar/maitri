import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# ==========================================================
# 1. DATASET PREPARATION
# ==========================================================
print("Preparing the FER-2013 dataset from image folders...")

data_dir = 'archive'  # path to main dataset folder

# Auto-detect number of classes
num_classes = len(os.listdir(os.path.join(data_dir, 'train')))
print(f"Detected {num_classes} classes.")

# Image augmentation to improve generalization
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Load training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_dir, 'train'),
    labels='inferred',
    label_mode='categorical',
    image_size=(48, 48),
    color_mode='grayscale',
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    subset='training',
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_dir, 'train'),
    labels='inferred',
    label_mode='categorical',
    image_size=(48, 48),
    color_mode='grayscale',
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    subset='validation',
    seed=123
)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("Dataset loaded and ready.")

# ==========================================================
# 2. MODEL ARCHITECTURE
# ==========================================================
print("Building the CNN model...")
model = Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(48,48,1)),  # normalize pixels
    data_augmentation,
    
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model built successfully.")

# ==========================================================
# 3. TRAINING THE MODEL
# ==========================================================
print("Training the model...")

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('emotion_model.h5', save_best_only=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,  # faster for hackathon, can increase if needed
    callbacks=[early_stopping, model_checkpoint]
)

print("Training finished. Model saved as 'emotion_model.h5'")
