import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import requests
import os
import io

# -----------------------------
# Dropbox Model URL
# -----------------------------
MODEL_PATH = "oil_spill_model_deploy.pth"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/stl47n6ixrzv59xs2jt4m/oil_spill_model_deploy.pth?rlkey=rojyk0fq73mk8tai8jc3exrev&dl=1"

# -----------------------------
# ResNet34-based UNet Implementation
# -----------------------------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNetUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResNetUNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Encoder (ResNet-like layers)
        self.encoder1 = self._make_layer(64, 64, 3)
        self.encoder2 = self._make_layer(64, 128, 4, stride=2)
        self.encoder3 = self._make_layer(128, 256, 6, stride=2)
        self.encoder4 = self._make_layer(256, 512, 3, stride=2)
        
        # Decoder (simple upsampling)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
        
    def _make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(blocks):
            layers.append(BasicBlock(planes, planes))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        
        # Decoder
        x = self.decoder(x)
        return x

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

        # Create model
        model = ResNetUNet()
        
        # Try to load weights with flexible loading
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Try to load state dict with different strategies
        try:
            model.load_state_dict(state_dict, strict=True)
            st.success("✅ Model loaded successfully!")
        except:
            # If strict loading fails, try flexible loading
            st.warning("⚠️ Using flexible model loading...")
            model_dict = model.state_dict()
            
            # Filter out unnecessary keys
            pretrained_dict = {k: v for k, v in state_dict.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            
            # Load what we can
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            st.success("✅ Model loaded (partial weights)")
        
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
        # Resize image to 256x256
        image_resized = image.resize((256, 256))
        img_array = np.array(image_resized).astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std

        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
        return img_tensor, image_resized
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {e}")
        return None, None

# -----------------------------
# Postprocess Prediction
# -----------------------------
def postprocess_prediction(prediction, original_size, confidence_threshold=0.5):
    try:
        # Convert prediction to numpy
        pred_mask = prediction.squeeze().cpu().numpy()
        
        # Apply confidence threshold
        binary_mask = (pred_mask > confidence_threshold).astype(np.uint8) * 255
        
        # Resize mask back to original image size
        mask_image = Image.fromarray(binary_mask)
        mask_resized = mask_image.resize(original_size, Image.NEAREST)
        
        return mask_resized, pred_mask
    except Exception as e:
        st.error(f"❌ Error postprocessing prediction: {e}")
        return None, None

# -----------------------------
# Create Overlay
# -----------------------------
def create_overlay(original_image, mask):
    try:
        # Convert images to numpy arrays
        original_np = np.array(original_image)
        mask_np = np.array(mask)
        
        # Create overlay (red for oil spills)
        overlay = original_np.copy()
        overlay[mask_np > 0] = [255, 0, 0]  # Red color for oil spills
        
        # Blend overlay with original (50% transparency)
        alpha = 0.5
        blended = (original_np * (1 - alpha) + overlay * alpha).astype(np.uint8)
        
        return Image.fromarray(blended)
    except Exception as e:
        st.error(f"❌ Error creating overlay: {e}")
        return original_image

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊", layout="wide")
st.title("🌊 Oil Spill Detection AI")
st.write("Upload a satellite image to detect oil spills using deep learning.")

with st.sidebar:
    st.header("⚙️ Detection Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
    st.header("ℹ️ About")
    st.write("This app uses a custom ResNet-UNet model to detect oil spills in satellite imagery.")

# Initialize model
if 'model' not in st.session_state:
    with st.spinner("🔄 Loading AI model..."):
        model, device = load_model()
        st.session_state.model = model
        st.session_state.device = device

# Upload image
uploaded_file = st.file_uploader("📤 Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load original image
        original_image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(original_image, use_column_width=True)
        
        if st.session_state.model is None:
            st.error("❌ Model failed to load. Using demo mode.")
            # Demo mode with simulated detection
            if st.button("🎯 Simulate Detection"):
                st.info("This is a demo. In full version, AI would detect oil spills.")
        else:
            if st.button("🎯 Detect Oil Spills", type="primary"):
                with st.spinner("🔄 Analyzing image for oil spills..."):
                    # Preprocess image
                    input_tensor, processed_image = preprocess_image(original_image)
                    
                    if input_tensor is not None:
                        # Move to device and predict
                        input_tensor = input_tensor.to(st.session_state.device)
                        
                        with torch.no_grad():
                            output = st.session_state.model(input_tensor)
                            prediction = torch.sigmoid(output)
                        
                        # Postprocess prediction
                        pred_mask, confidence_map = postprocess_prediction(
                            prediction, original_image.size, confidence_threshold
                        )
                        
                        if pred_mask is not None:
                            # Create overlay
                            overlay_image = create_overlay(original_image, pred_mask)
                            
                            with col2:
                                st.subheader("🔍 Detection Results")
                                st.image(overlay_image, 
                                       caption="Oil Spill Detection (Red = Oil Spill)", 
                                       use_column_width=True)
                            
                            # Calculate metrics
                            mask_array = np.array(pred_mask)
                            spill_pixels = np.sum(mask_array > 0)
                            total_pixels = mask_array.size
                            spill_percentage = (spill_pixels / total_pixels) * 100
                            
                            # Display metrics
                            st.subheader("📊 Detection Metrics")
                            metric_col1, metric_col2 = st.columns(2)
                            
                            with metric_col1:
                                st.metric("Spill Area", f"{spill_percentage:.2f}%")
                            
                            with metric_col2:
                                status = "🔴 Spill Detected" if spill_percentage > 0.1 else "🟢 No Spill"
                                st.metric("Status", status)
                            
                            # Download button
                            st.subheader("💾 Download Results")
                            mask_buffer = io.BytesIO()
                            pred_mask.save(mask_buffer, format="PNG")
                            st.download_button(
                                label="Download Prediction Mask",
                                data=mask_buffer.getvalue(),
                                file_name="oil_spill_mask.png",
                                mime="image/png"
                            )
                            
                            st.balloons()
                            
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

else:
    st.info("👆 Please upload a satellite image to begin oil spill detection")
