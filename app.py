import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('celeb_model.h5')

# List of celebrity names
celebrities = ['Lionel Messi', 'Maria Sharapova', 'Roger Federer','Serena willams','Virat Kohili']  # Replace with actual celebrity names

# Streamlit app
st.title("Celebrity Image Classification")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for model prediction
    image = image.resize((128, 128))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.utils.normalize(image_array, axis=1)

    # Make prediction using the loaded model
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    predicted_celebrity = celebrities[predicted_class]

    # Button to trigger prediction
    if st.button("Predict Celebrity"):
        st.subheader("Prediction:")
        st.write(f"The model predicts that the celebrity in the image is: {predicted_celebrity}")
