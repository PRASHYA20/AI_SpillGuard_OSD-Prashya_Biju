import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import io
import requests
import os

# -----------------------------
# Model Configuration
# -----------------------------
MODEL_PATH = "oil_spill_model_deploy.pth"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/stl47n6ixrzv59xs2jt4m/oil_spill_model_deploy.pth?rlkey=rojyk0fq73mk8tai8jc3exrev&dl=1"

# -----------------------------
# Create UNet Model
# -----------------------------
def create_unet_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
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
            downloaded = 0
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(progress)
            progress_bar.empty()
            st.success("✅ Model downloaded!")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"🖥️ Using device: {device}")

    if not download_model():
        return None, device

    try:
        model = create_unet_model()
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Remove DataParallel prefix if exists
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
# Preprocess Image
# -----------------------------
def preprocess_image(image):
    image_resized = image.resize((256, 256))
    img_array = np.array(image_resized).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    # Float32 tensor
    img_tensor = torch.from_numpy(img_array).permute(2,0,1).unsqueeze(0).float()
    return img_tensor, image_resized

# -----------------------------
# Streamlit App - EXACT SAME FORMAT
# -----------------------------
st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊", layout="wide")
st.title("🌊 Oil Spill Segmentation with UNet (ResNet34)")
st.write("Upload a satellite image to detect oil spills.")

with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.1)
    st.header("ℹ️ About")
    st.write("This app uses a UNet model (ResNet34 backbone) for oil spill segmentation.")

# Initialize model
if 'model' not in st.session_state:
    with st.spinner("🔄 Loading model..."):
        model, device = load_model()
        st.session_state.model = model
        st.session_state.device = device

# Upload image
uploaded_file = st.file_uploader("📤 Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display Original Image at the top
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    if st.session_state.model is None:
        st.error("❌ Model failed to load. Please check weights.")
    else:
        with st.spinner("🔄 Predicting..."):
            input_tensor, processed_image = preprocess_image(image)
            input_tensor = input_tensor.to(st.session_state.device, dtype=torch.float32)

            with torch.no_grad():
                output = st.session_state.model(input_tensor)
                prediction = torch.sigmoid(output).squeeze().cpu().numpy()

            # Binary mask
            binary_mask = (prediction > confidence_threshold).astype(np.uint8) * 255

            # Create two columns for side-by-side display
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Processed Image")
                st.image(processed_image, use_column_width=True)
            
            with col2:
                st.subheader("Oil Spill Overlay")
                # Overlay mask on original - RED for oil spill
                overlay = processed_image.copy()
                overlay_np = np.array(overlay)
                overlay_np[binary_mask > 0] = [255, 0, 0]  # red for oil spill
                overlay_img = Image.fromarray(overlay_np)
                st.image(overlay_img, use_column_width=True)

            # Metrics - EXACT SAME as your working version
            spill_area = np.sum(binary_mask > 0) / (binary_mask.shape[0] * binary_mask.shape[1]) * 100
            max_conf = np.max(prediction) * 100

            st.subheader("📊 Detection Results")
            
            # Create metrics in columns
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric("Spill Area", f"{spill_area:.2f}%")
            with col4:
                st.metric("Max Confidence", f"{max_conf:.1f}%")
            with col5:
                status = "🔴 Spill Detected" if spill_area > 1.0 else "🟢 No Spill"
                st.metric("Status", status)

            # Download mask - EXACT SAME as your working version
            mask_image = Image.fromarray(binary_mask)
            buf = io.BytesIO()
            mask_image.save(buf, format="PNG")
            st.download_button(
                label="💾 Download Prediction Mask",
                data=buf.getvalue(),
                file_name="oil_spill_mask.png",
                mime="image/png"
            )

else:
    st.info("👆 Please upload a satellite image to begin detection.")
