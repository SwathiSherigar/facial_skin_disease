import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the saved model
model = load_model(r'A:\face_disease\FSD\skin_disease_model.h5')

# Define class labels
classes = ['Acne', 'Actinic Keratosis', 'Basal Cell Carcinom', 'Eczemaa', 'Rosacea']

# App title
st.title("Skin Disease Classification")
st.write("Upload an image of a skin condition to predict its class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    input_shape = (150, 150)  # Input shape of the model
    test_image = image.resize(input_shape)
    test_image_array = np.array(test_image)
    test_image_array = np.expand_dims(test_image_array, axis=0)
    test_image_array = test_image_array / 255.0

    # Make prediction
    predictions = model.predict(test_image_array)
    predicted_class = np.argmax(predictions)
    predicted_label = classes[predicted_class]

    # Display the prediction
    st.write(f"**Predicted Class:** {predicted_label}")
    st.bar_chart(predictions[0])
