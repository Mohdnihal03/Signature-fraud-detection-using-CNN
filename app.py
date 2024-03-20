import tensorflow as tf
from tensorflow import keras 
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os

# Constants
IMAGE_SIZE = (128, 128)
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 20

# Function to load images and labels
def load_data(data_dir):
    images = []
    labels = []
    for folder_name in os.listdir(data_dir):
        if folder_name == 'genuine':
            label = 1
        elif folder_name == 'forged':
            label = 0
        else:
            continue
        folder_path = os.path.join(data_dir, folder_name)
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image = image.resize(IMAGE_SIZE)
            images.append(np.array(image))
            labels.append(label)
    return np.array(images), np.array(labels)

# Load data
data_dir = 'C:/Users/nihall/Documents/e signature/cheque images'
images, labels = load_data(data_dir)

# Normalize pixel values
images = images.astype('float32') / 255.0

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Reshape the input images to have a rank of 4
X_train = X_train.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
X_test = X_test.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

# Save the model
model.save("signature_detection.h5")
