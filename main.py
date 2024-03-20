import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the pre-trained model
model = load_model('signature_detection.h5')

# Define class labels (0 for forged, 1 for genuine)
class_labels = {1: 'genuine', 0: 'forged'}

# Function to process the uploaded image
def process_image(image):
    # Convert the image to grayscale
    image_gray = image.convert('L')
    # Resize the grayscale image to match the input size of your model
    image_gray = image_gray.resize((128, 128))
    # Convert the grayscale image to numpy array and normalize the pixel values
    image_array = np.array(image_gray) / 255.0
    return image_array
# Streamlit app
def main():
    st.title('Signature Detection App')

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction when button is clicked
        if st.button('Predict'):
            # Process the uploaded image
            processed_image = process_image(image)
            # Make prediction
            prediction = model.predict(np.array([processed_image]))
            # Get the predicted class label
            predicted_class = np.argmax(prediction)
            # Display the prediction
            if predicted_class == 1:
                st.write("Signature is genuine")
            else:
                st.write("Signature is forged")

if __name__ == '__main__':
    main()
