import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import io
import cv2
import segmentation_models_pytorch as smp

# -----------------------------
# Dropbox Model URL
# -----------------------------
MODEL_PATH = "oil_spill_model_deploy.pth"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/stl47n6ixrzv59xs2jt4m/oil_spill_model_deploy.pth?rlkey=rojyk0fq73mk8tai8jc3exrev&st=w6qm08lh&dl=1"

# -----------------------------
# Create SMP UNet (ResNet34 backbone)
# -----------------------------
def create_unet_model():
    model = smp.Unet(
        encoder_name="resnet34",   # Pre-trained encoder
        encoder_weights=None,      # no need for imagenet in inference
        in_channels=3,             # RGB input
        classes=1,                 # binary segmentation
        activation=None,           # we’ll use sigmoid manually
    )
    return model

# -----------------------------
# Download Model
# -----------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("🔽 Downloading model from Dropbox...")
        try:
            response = requests.get(DROPBOX_URL, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with open(MODEL_PATH, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(progress)
                            status_text.text(f"Downloaded: {downloaded}/{total_size} bytes ({progress:.1%})")
            
            progress_bar.empty()
            status_text.empty()
            st.success("✅ Model downloaded successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            return False
    return True

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    st.write(f"🖥️ Using device: {device}")

    if not download_model():
        return None, device

    try:
        model = create_unet_model()

        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Remove "module." prefix if saved with DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        st.success("✅ Model loaded successfully!")
        return model, device

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, device

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image):
    image_resized = image.resize((256, 256))
    img_array = np.array(image_resized).astype(np.float32) / 255.0

    # ImageNet normalization (same as training)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor, image_resized

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Oil Spill Detection - UNet", page_icon="🌊", layout="wide")
st.title("🌊 Oil Spill Segmentation with UNet (ResNet34)")
st.write("Upload a satellite image to detect oil spills using deep learning.")

with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1, max_value=0.9,
        value=0.5, step=0.1,
        help="Adjust the sensitivity of spill detection"
    )
    st.header("ℹ️ About")
    st.write("This app uses a UNet model (ResNet34 backbone) trained for oil spill segmentation.")

# Initialize model
if 'model' not in st.session_state:
    with st.spinner("🔄 Loading UNet model..."):
        model, device = load_model()
        st.session_state.model = model
        st.session_state.device = device

# File upload
uploaded_file = st.file_uploader("📤 Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
        st.write(f"Original size: {image.size}")
    
    with col2:
        if st.session_state.model is None:
            st.error("❌ Model failed to load. Please check weights.")
        else:
            with st.spinner("🔄 Processing image..."):
                input_tensor, processed_image = preprocess_image(image)
                input_tensor = input_tensor.to(st.session_state.device)

                with torch.no_grad():
                    output = st.session_state.model(input_tensor)
                    prediction = torch.sigmoid(output).squeeze().cpu().numpy()

                binary_mask = (prediction > confidence_threshold).astype(np.uint8) * 255

                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                ax1.imshow(processed_image)
                ax1.set_title("Processed Image")
                ax1.axis('off')
                ax2.imshow(prediction, cmap='viridis')
                ax2.set_title("Probability Map")
                ax2.axis('off')
                ax3.imshow(processed_image)
                ax3.imshow(binary_mask, cmap='Reds', alpha=0.5)
                ax3.set_title("Oil Spill Detection")
                ax3.axis('off')
                st.pyplot(fig)

            spill_area = np.sum(binary_mask > 0) / (binary_mask.shape[0] * binary_mask.shape[1]) * 100
            max_confidence = np.max(prediction) * 100

            st.subheader("📊 Detection Results")
            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric("Spill Area", f"{spill_area:.2f}%")
            with col4:
                st.metric("Max Confidence", f"{max_confidence:.1f}%")
            with col5:
                status = "🔴 Spill Detected" if spill_area > 1.0 else "🟢 No Spill"
                st.metric("Status", status)

            mask_image = Image.fromarray(binary_mask)
            buf = io.BytesIO()
            mask_image.save(buf, format="PNG")
            st.download_button(
                label="💾 Download Prediction Mask",
                data=buf.getvalue(),
                file_name="oil_spill_mask.png",
                mime="image/png",
                use_container_width=True
            )
else:
    st.info("👆 Please upload a satellite image to begin detection.")
