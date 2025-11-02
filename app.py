import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

#Page title
st.title("Chihuahua🐕 or Muffin🧁")
st.write("Upload an image and the model will classify it as Chihuahua or Muffin.")

#Load your trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("bin_classifier.h5")
    return model

model = load_model()

#Upload image
uploaded_file = st.file_uploader("Choose an image (.jpg/.png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "Muffin" if prediction > 0.5 else "Chihuahua"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### ✅ Prediction: **{label}**")
    st.progress(float(confidence))
    st.write(f"Confidence: {confidence*100:.2f}%")
