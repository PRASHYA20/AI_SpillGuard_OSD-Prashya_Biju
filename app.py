import os
os.environ['TIMM_DISABLE_TEARDOWN'] = '1'

import streamlit as st
import traceback

try:
    # Rest of your imports and code
    import torch
    import segmentation_models_pytorch as smp
    # ... rest of your app
except Exception as e:
    st.error(f"Error: {e}")import streamlit as st
import traceback
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import io
import requests
import os
import matplotlib.pyplot as plt

# -----------------------------
# Dropbox Model URL
# -----------------------------
MODEL_PATH = "oil_spill_model_deploy.pth"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/stl47n6ixrzv59xs2jt4m/oil_spill_model_deploy.pth?rlkey=rojyk0fq73mk8tai8jc3exrev&dl=1"

# -----------------------------
# Create UNet Model
# -----------------------------
def create_unet_model():
    try:
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
        return model
    except Exception as e:
        st.error(f"❌ Error creating model: {e}")
        return None

# -----------------------------
# Download model if missing
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
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write(f"🖥️ Using device: {device}")

        if not download_model():
            return None, device

        model = create_unet_model()
        if model is None:
            return None, device

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
        st.code(traceback.format_exc())
        return None, device

# -----------------------------
# Preprocess Image
# -----------------------------
def preprocess_image(image):
    try:
        image_resized = image.resize((256, 256))
        img_array = np.array(image_resized).astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std

        # Float32 tensor
        img_tensor = torch.from_numpy(img_array).permute(2,0,1).unsqueeze(0).float()
        return img_tensor, image_resized
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {e}")
        return None, None

# -----------------------------
# Streamlit App
# -----------------------------
try:
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
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Original Image", use_column_width=True)

            if st.session_state.model is None:
                st.error("❌ Model failed to load. Please check weights.")
            else:
                with st.spinner("🔄 Predicting..."):
                    input_tensor, processed_image = preprocess_image(image)
                    
                    if input_tensor is None:
                        st.error("❌ Failed to preprocess image")
                    else:
                        input_tensor = input_tensor.to(st.session_state.device, dtype=torch.float32)

                        with torch.no_grad():
                            output = st.session_state.model(input_tensor)
                            prediction = torch.sigmoid(output).squeeze().cpu().numpy()

                        # Binary mask
                        binary_mask = (prediction > confidence_threshold).astype(np.uint8) * 255

                        # Overlay mask on original
                        overlay = processed_image.copy()
                        overlay_np = np.array(overlay)
                        overlay_np[binary_mask>0] = [255,0,0]  # red for oil spill
                        overlay_img = Image.fromarray(overlay_np)

                        col1, col2 = st.columns(2)
                        col1.image(processed_image, caption="Processed Image", use_column_width=True)
                        col2.image(overlay_img, caption="Oil Spill Overlay", use_column_width=True)

                        # Metrics
                        spill_area = np.sum(binary_mask>0) / (binary_mask.shape[0]*binary_mask.shape[1]) * 100
                        max_conf = np.max(prediction) * 100

                        st.subheader("📊 Detection Results")
                        st.metric("Spill Area", f"{spill_area:.2f}%")
                        st.metric("Max Confidence", f"{max_conf:.1f}%")
                        status = "🔴 Spill Detected" if spill_area > 1.0 else "🟢 No Spill"
                        st.metric("Status", status)

                        # Download mask
                        mask_image = Image.fromarray(binary_mask)
                        buf = io.BytesIO()
                        mask_image.save(buf, format="PNG")
                        st.download_button(
                            label="💾 Download Prediction Mask",
                            data=buf.getvalue(),
                            file_name="oil_spill_mask.png",
                            mime="image/png"
                        )
        except Exception as e:
            st.error(f"❌ Error processing uploaded image: {e}")
            st.code(traceback.format_exc())
    else:
        st.info("👆 Please upload a satellite image to begin detection.")

except Exception as e:
    st.error(f"🚨 Critical App Error: {e}")
    st.code(traceback.format_exc())

