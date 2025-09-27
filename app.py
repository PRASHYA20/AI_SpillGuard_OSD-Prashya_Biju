import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests
import os
import io

# -----------------------------
# Dropbox Model URL
# -----------------------------
MODEL_PATH = "oil_spill_model_deploy.pth"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/stl47n6ixrzv59xs2jt4m/oil_spill_model_deploy.pth?rlkey=rojyk0fq73mk8tai8jc3exrev&st=w6qm08lh&dl=1"

# -----------------------------
# Define your UNet model
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(UNet, self).__init__()
        self.dc1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.dc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.dc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.dc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.dc5 = DoubleConv(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dc6 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dc7 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dc8 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dc9 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        x1 = self.dc1(x)
        x2 = self.dc2(self.pool1(x1))
        x3 = self.dc3(self.pool2(x2))
        x4 = self.dc4(self.pool3(x3))
        x5 = self.dc5(self.pool4(x4))
        x = self.up1(x5)
        x = self.dc6(torch.cat([x, x4], dim=1))
        x = self.up2(x)
        x = self.dc7(torch.cat([x, x3], dim=1))
        x = self.up3(x)
        x = self.dc8(torch.cat([x, x2], dim=1))
        x = self.up4(x)
        x = self.dc9(torch.cat([x, x1], dim=1))
        x = self.out_conv(x)
        return x

# -----------------------------
# Download Model if not exists
# -----------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("🔽 Downloading model from Dropbox...")
        try:
            r = requests.get(DROPBOX_URL, allow_redirects=True, timeout=60)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"🖥️ Using device: {device}")
    
    if not download_model():
        return None, device
    
    try:
        model = UNet(in_ch=3, out_ch=1)
        # Load with map_location to handle CPU/GPU compatibility
        state_dict = torch.load(MODEL_PATH, map_location=torch.device(device))
        
        # Handle state dict format
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        st.success("✅ Model loaded successfully!")
        return model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, device

# -----------------------------
# Simple preprocessing without torchvision
# -----------------------------
def preprocess_image(image):
    # Resize
    image = image.resize((256, 256))
    # Convert to numpy array
    img_array = np.array(image).astype(np.float32) / 255.0
    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Oil Spill Segmentation (UNet)")
st.write("Upload a satellite image to detect possible oil spills using UNet architecture.")

# Initialize model
if 'model_loaded' not in st.session_state:
    with st.spinner("Loading UNet model..."):
        model, device = load_model()
        st.session_state.model = model
        st.session_state.device = device
        st.session_state.model_loaded = True
else:
    model = st.session_state.model
    device = st.session_state.device

uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    
    if model is None:
        st.error("❌ Model failed to load. Please check the console for errors.")
    else:
        # Preprocess
        input_tensor = preprocess_image(image).to(device)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()

        # Apply threshold
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.1)
        mask = (pred > confidence_threshold).astype(np.uint8) * 255

        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis("off")
        
        # Prediction mask
        ax2.imshow(mask, cmap="hot")
        ax2.set_title("Prediction Mask")
        ax2.axis("off")
        
        # Overlay
        overlay = np.array(image.resize((256, 256)))
        ax3.imshow(overlay)
        ax3.imshow(mask, cmap="Reds", alpha=0.5)
        ax3.set_title("Overlay (Red = Oil Spill)")
        ax3.axis("off")
        
        plt.tight_layout()
        
        with col2:
            st.pyplot(fig)
        
        # Statistics
        spill_area = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1]) * 100
        
        st.subheader("📊 Detection Statistics")
        st.write(f"Spill Area: {spill_area:.2f}%")
        st.write(f"Threshold: {confidence_threshold}")
        
        # Download Mask Button
        mask_img = Image.fromarray(mask.astype(np.uint8))
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="💾 Download Predicted Mask",
            data=byte_im,
            file_name="oil_spill_mask.png",
            mime="image/png"
        )
