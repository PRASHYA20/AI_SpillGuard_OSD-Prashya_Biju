import streamlit as st
from PIL import Image
import numpy as np
import requests
import os

# -----------------------------
# Dropbox Model URL
# -----------------------------
MODEL_PATH = "oil_spill_model_deploy.pth"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/stl47n6ixrzv59xs2jt4m/oil_spill_model_deploy.pth?rlkey=rojyk0fq73mk8tai8jc3exrev&dl=1"

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊", layout="wide")
st.title("🌊 Oil Spill Detection App")
st.success("✅ App loaded successfully!")

with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.1)
    st.header("ℹ️ About")
    st.write("Upload satellite images for oil spill detection.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Simulate processing (replace with actual model later)
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                # Simple image analysis (replace with ML model)
                img_array = np.array(image)
                st.write(f"Image size: {img_array.shape}")
                
                # Simulate detection results
                st.success("Analysis complete!")
                st.metric("Status", "Ready for ML Integration")
                st.info("ML model integration pending - app is stable and ready!")
                
    except Exception as e:
        st.error(f"Error processing image: {e}")

else:
    st.info("👆 Please upload a satellite image")

# Model download section (optional)
if st.checkbox("Show model download status"):
    if os.path.exists(MODEL_PATH):
        st.success("✅ Model file is available")
    else:
        st.warning("⚠️ Model file not downloaded yet")
        if st.button("Download Model"):
            try:
                response = requests.get(DROPBOX_URL, stream=True)
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("✅ Model downloaded successfully!")
            except Exception as e:
                st.error(f"Download failed: {e}")
