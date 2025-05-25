import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np

# Set page configuration for a wider layout and custom title
st.set_page_config(
    page_title="Flower Classification App",
    
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    /* Header styling */
    .stHeader {
        color: #2e7d32;
        font-family: 'Arial', sans-serif;
        text-align: center;
        margin-bottom: 20px;
    }
    /* Button styling */
    .stButton>button {
        background-color: #4caf50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    /* Image container styling */
    .stImage {
        border: 2px solid #ddd;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    /* Result text styling */
    .result-text {
        font-size: 18px;
        color: #1e88e5;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with additional information
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=Flower+App", caption="Flower Classification", use_container_width=True)
    st.markdown("### About This App")
    st.markdown("Upload an image of a flower, and this app will classify it as one of the following: Lilly, Lotus, Orchid, Sunflower, or Tulip using a pre-trained CNN model.")
    st.markdown("#### Instructions")
    st.markdown("- Upload a flower image (JPG, PNG, etc.).")
    st.markdown("- Wait for the model to predict the flower type.")
    st.markdown("- View the result with a confidence score.")
    st.markdown("#### Model Details")
    st.markdown("- **Model**: Convolutional Neural Network (CNN)")
    st.markdown("- **Classes**: Lilly, Lotus, Orchid, Sunflower, Tulip")
    st.markdown("- **Input Size**: 180x180 pixels")

# Main content
st.markdown("<h1 class='stHeader'>🌸 Flower Classification CNN Model 🌸</h1>", unsafe_allow_html=True)

# Flower names
flower_names = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']

# Load the model
@st.cache_resource
def load_flower_model():
    return load_model('Flower_Recog_Model.h5')

model = load_flower_model()

# Function to classify images
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = f'The Image belongs to **{flower_names[np.argmax(result)]}** with a score of **{np.max(result)*100:.2f}%**'
    return outcome

# File uploader
st.markdown("#### Upload a Flower Image")
uploaded_file = st.file_uploader("Choose an image file (JPG, PNG, etc.)", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

# Ensure the 'upload' directory exists
if not os.path.exists('upload'):
    os.makedirs('upload')

# Process uploaded file
if uploaded_file is not None:
    # Save the uploaded file
    file_path = os.path.join('upload', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Flower Image", width=200, output_format="auto")

    # Show a button to trigger classification
    if st.button("Classify Image"):
        with st.spinner("Classifying..."):
            result = classify_images(file_path)
            st.markdown(f"<div class='result-text'>{result}</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Developed with Streamlit | Powered by TensorFlow & Keras</p>", unsafe_allow_html=True)