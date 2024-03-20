import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("signature_detection.h5")

# Function to preprocess the image
def preprocess_image(image_path, target_size=(128, 128)):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0  # Normalize pixel values
    return image_array

# Provide the path to your image
image_path = "C:/Users/nihall/Documents/e signature/cheque images/genuine/002002_000.png"

# Preprocess the image
input_image = preprocess_image(image_path)

# Add batch dimension as the model expects batches of images
input_image = np.expand_dims(input_image, axis=0)

# Perform prediction
predictions = model.predict(input_image)

# Assuming binary classification, get the predicted class
predicted_class = "Genuine" if predictions[0][0] > 0.5 else "Forged"

# Print the prediction
print("Predicted class:", predicted_class)
