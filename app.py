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

# -----------------------------
# Dropbox Model URL
# -----------------------------
MODEL_PATH = "oil_spill_model_deploy.pth"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/stl47n6ixrzv59xs2jt4m/oil_spill_model_deploy.pth?rlkey=rojyk0fq73mk8tai8jc3exrev&st=w6qm08lh&dl=1"

# -----------------------------
# UNet Model Definition
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
    # Use CPU for compatibility
    device = torch.device('cpu')
    st.write(f"🖥️ Using device: {device}")
    
    if not download_model():
        return None, device
    
    try:
        model = UNet(in_ch=3, out_ch=1)
        
        # Load model with error handling for state dict format
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        # Handle different state dict formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
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
        st.info("Trying alternative loading method...")
        
        try:
            # Alternative: Load directly without state dict processing
            model = UNet(in_ch=3, out_ch=1)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            model.eval()
            st.success("✅ Model loaded with alternative method!")
            return model, device
        except Exception as e2:
            st.error(f"❌ Alternative loading failed: {e2}")
            return None, device

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image):
    """Preprocess image for UNet model"""
    # Resize to 256x256
    image_resized = image.resize((256, 256))
    
    # Convert to numpy and normalize
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, image_resized

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Oil Spill Detection - UNet",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Oil Spill Segmentation with UNet")
st.write("Upload a satellite image to detect oil spills using deep learning.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Adjust the sensitivity of spill detection"
    )
    
    st.header("ℹ️ About")
    st.write("This app uses a UNet model trained for oil spill segmentation in satellite imagery.")

# Initialize model
if 'model' not in st.session_state:
    with st.spinner("🔄 Loading UNet model..."):
        model, device = load_model()
        st.session_state.model = model
        st.session_state.device = device

# File upload
uploaded_file = st.file_uploader("📤 Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
        st.write(f"Original size: {image.size}")
    
    with col2:
        if st.session_state.model is None:
            st.error("❌ Model failed to load. Using fallback detection.")
            
            # Fallback: Simple color-based detection
            st.info("🔄 Using fallback detection method...")
            img_array = np.array(image.resize((256, 256)))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            _, simple_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            
            # Display fallback results
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].imshow(img_array)
            ax[0].set_title("Original")
            ax[0].axis('off')
            
            ax[1].imshow(simple_mask, cmap='hot')
            ax[1].set_title("Fallback Detection")
            ax[1].axis('off')
            
            st.pyplot(fig)
            
        else:
            # Use UNet model for prediction
            with st.spinner("🔄 Processing image with UNet..."):
                # Preprocess
                input_tensor, processed_image = preprocess_image(image)
                input_tensor = input_tensor.to(st.session_state.device)
                
                # Prediction
                with torch.no_grad():
                    output = st.session_state.model(input_tensor)
                    prediction = torch.sigmoid(output).squeeze().cpu().numpy()
                
                # Apply threshold
                binary_mask = (prediction > confidence_threshold).astype(np.uint8) * 255
                
                # Create visualization
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original
                ax1.imshow(processed_image)
                ax1.set_title("Processed Image")
                ax1.axis('off')
                
                # Probability map
                ax2.imshow(prediction, cmap='viridis')
                ax2.set_title("Probability Map")
                ax2.axis('off')
                
                # Binary mask overlay
                ax3.imshow(processed_image)
                ax3.imshow(binary_mask, cmap='Reds', alpha=0.5)
                ax3.set_title("Oil Spill Detection")
                ax3.axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Statistics
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
            
            # Download mask
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
    
    # Sample section
    with st.expander("📋 How it works"):
        st.markdown("""
        **UNet Oil Spill Detection:**
        1. **Upload** a satellite image of a water body
        2. **UNet model** processes the image pixel-by-pixel
        3. **Probability map** shows confidence levels
        4. **Binary mask** highlights detected spills
        5. **Download** the prediction mask for further analysis
        
        **Model Architecture:**
        - Encoder-decoder structure with skip connections
        - Trained on satellite imagery
        - Outputs pixel-wise spill probabilities
        """)

st.markdown("---")
st.markdown("*Oil Spill Detection using UNet Deep Learning Model*")
