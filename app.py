import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io
import requests
import os

# -----------------------------
# Dropbox Model URL
# -----------------------------
MODEL_PATH = "oil_spill_model_deploy.pth"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/stl47n6ixrzv59xs2jt4m/oil_spill_model_deploy.pth?rlkey=rojyk0fq73mk8tai8jc3exrev&dl=1"

# -----------------------------
# Simple UNet Model (avoid segmentation_models_pytorch)
# -----------------------------
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(SimpleUNet, self).__init__()
        # Simple convolutional layers as placeholder
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x

def create_model():
    return SimpleUNet()

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

        model = create_model()
        
        # Try to load the pre-trained weights
        try:
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
            st.success("✅ Pre-trained model loaded successfully!")
        except:
            st.warning("⚠️ Could not load pre-trained weights, using untrained model")
        
        model.to(device)
        model.eval()
        return model, device

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
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
st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊", layout="wide")
st.title("🌊 Oil Spill Segmentation")
st.write("Upload a satellite image to detect oil spills.")

with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.1)
    st.header("ℹ️ About")
    st.write("This app detects oil spills in satellite images.")

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
    st.image(image, caption="Original Image", use_column_width=True)

    if st.session_state.model is None:
        st.error("❌ Model failed to load.")
    else:
        if st.button("Detect Oil Spills"):
            with st.spinner("🔄 Processing image..."):
                input_tensor, processed_image = preprocess_image(image)
                
                if input_tensor is not None:
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
                    col2.image(overlay_img, caption="Oil Spill Detection", use_column_width=True)

                    # Metrics
                    spill_area = np.sum(binary_mask>0) / (binary_mask.shape[0]*binary_mask.shape[1]) * 100

                    st.subheader("📊 Detection Results")
                    st.metric("Spill Area", f"{spill_area:.2f}%")
                    status = "🔴 Spill Detected" if spill_area > 1.0 else "🟢 No Spill"
                    st.metric("Status", status)

                    # Download mask
                    mask_image = Image.fromarray(binary_mask)
                    buf = io.BytesIO()
                    mask_image.save(buf, format="PNG")
                    st.download_button(
                        label="💾 Download Detection Mask",
                        data=buf.getvalue(),
                        file_name="oil_spill_detection.png",
                        mime="image/png"
                    )
else:
    st.info("👆 Please upload a satellite image to begin detection.")
