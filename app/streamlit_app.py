import streamlit as st
import numpy as np
import tensorflow as tf
from explainability.gradcam import make_gradcam_heatmap, display_gradcam

st.set_page_config(layout="wide")
st.title("Interpretable Colorectal Cancer Histopathology Classifier")

st.markdown("""
This application visualizes **Grad-CAM explanations** for a deep-learning model
trained on colorectal cancer histopathology images, with a focus on model errors.
""")

st.info("Upload or select sample images to visualize model attention.")

st.warning("Demo app structure – connect model and data for full functionality.")
