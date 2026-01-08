import streamlit as st
import numpy as np
import tensorflow as tf
from explainability.gradcam import make_gradcam_heatmap, display_gradcam

st.set_page_config(layout="wide")
st.title("Interpretable Colorectal Cancer Histopathology Classifier")

uploaded = st.file_uploader("Upload a histopathology image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = tf.io.decode_image(uploaded.read(), channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image.numpy() / 255.0

    model = tf.keras.models.load_model("checkpoints/trained_model.h5")

    preds = model.predict(image[np.newaxis, ...])
    pred_class = np.argmax(preds[0])

    st.write(f"Predicted class index: {pred_class}")

    heatmap = make_gradcam_heatmap(
        image[np.newaxis, ...],
        model,
        "conv5_block3_out"
    )

    display_gradcam(image, heatmap)
