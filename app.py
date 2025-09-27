import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
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
        
        # Handle state dict format (it might be nested under 'model' key)
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
# Preprocessing
# -----------------------------
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

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

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.write("This app uses a custom UNet model to detect oil spills in satellite imagery.")
    st.write("**Instructions:**")
    st.write("1. Upload a satellite image (JPG, JPEG, PNG)")
    st.write("2. The UNet model will process the image")
    st.write("3. View the segmentation results")
    st.write("4. Download the mask if needed")
    
    st.header("Model Info")
    st.write(f"Framework: PyTorch {torch.__version__}")
    st.write("Architecture: Custom UNet")
    st.write("Input size: 256x256 pixels")
    
    # Confidence threshold slider
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Adjust the sensitivity of spill detection"
    )

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
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()

        # Apply threshold
        mask = (pred > confidence_threshold).astype(np.uint8) * 255

        # Create overlay
        overlay = np.array(image.resize((256, 256)))
        mask_resized = Image.fromarray(mask).resize((overlay.shape[1], overlay.shape[0]))
        mask_array = np.array(mask_resized)
        
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
        ax3.imshow(overlay)
        ax3.imshow(mask_array, cmap="Reds", alpha=0.5)
        ax3.set_title("Overlay (Red = Oil Spill)")
        ax3.axis("off")
        
        plt.tight_layout()
        
        with col2:
            st.pyplot(fig)
        
        # Statistics
        spill_area = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1]) * 100
        max_confidence = np.max(pred) * 100
        
        st.subheader("📊 Detection Statistics")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.metric("Spill Area Percentage", f"{spill_area:.2f}%")
        
        with col4:
            st.metric("Max Confidence", f"{max_confidence:.2f}%")
        
        with col5:
            status = "🟢 No Spill" if spill_area < 1.0 else "🔴 Spill Detected"
            st.metric("Status", status)
        
        # Download Mask Button
        mask_img = Image.fromarray(mask.astype(np.uint8))
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="💾 Download Predicted Mask",
            data=byte_im,
            file_name="oil_spill_mask.png",
            mime="image/png",
            use_container_width=True
        )

else:
    st.info("👆 Please upload a satellite image to get started.")

# Footer
st.markdown("---")
st.markdown("### How it works:")
st.markdown("""
- **UNet Architecture**: The model uses an encoder-decoder structure with skip connections
- **Segmentation**: Predicts pixel-wise probabilities for oil spill presence
- **Post-processing**: Applies threshold to create binary mask
- **Visualization**: Overlays detection results on original image
""")
