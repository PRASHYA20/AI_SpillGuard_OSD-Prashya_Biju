import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import io
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# Set page config
st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Oil Spill Detection")
st.write("Upload satellite imagery for AI-powered oil spill detection")

# Look for model files
def find_model_file():
    model_path = "oil_spill_model_deploy.pth"
    if os.path.exists(model_path):
        st.sidebar.success(f"✅ Model loaded: {os.path.getsize(model_path) / (1024 * 1024):.1f} MB")
        return model_path
    else:
        st.sidebar.error("❌ Model file not found")
        return None

# Define the model architecture (same as your local version)
class OilSpillSegmentationModel(nn.Module):
    def __init__(self):
        super(OilSpillSegmentationModel, self).__init__()
        self.encoder = models.resnet50(pretrained=False)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        
        self.segmentation_head = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        features = self.encoder(x)
        x = self.decoder(features)
        x = self.segmentation_head(x)
        x = self.sigmoid(x)
        return x

@st.cache_resource
def load_model():
    """Load the model"""
    model_path = find_model_file()
    
    if model_path is None:
        return None
    
    try:
        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Create model
        model = OilSpillSegmentationModel()
        
        # Load weights with strict=False
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        return model
        
    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {e}")
        return None

# Load model
model = load_model()

# SIMPLIFIED Settings - like your local version
st.sidebar.header("⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 0.99, 0.5, 0.01,  # Medium default
    help="Adjust detection sensitivity"
)

target_size = st.sidebar.selectbox(
    "Processing Size", [256, 512, 224], index=0,
    help="Image size for model processing"
)

# MINIMAL filtering - like your local version
apply_smoothing = st.sidebar.checkbox(
    "Apply Smoothing", value=True,
    help="Light smoothing to clean up edges"
)

def preprocess_for_model(image, target_size=(256, 256)):
    """Preprocess image for model inference"""
    if isinstance(image, Image.Image):
        original_pil = image.copy()
        original_array = np.array(image)
    else:
        original_pil = Image.fromarray(image)
        original_array = image.copy()
    
    original_h, original_w = original_array.shape[:2]
    
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(original_pil).unsqueeze(0)
    return image_tensor, original_array, (original_h, original_w)

def process_model_output(prediction, original_shape, confidence_threshold=0.5):
    """Simple processing like your local version"""
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.squeeze().detach().cpu().numpy()
    
    # Simple thresholding - no complex filtering
    binary_mask = (prediction > confidence_threshold).astype(np.uint8)
    
    # Resize to original dimensions
    mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
    mask_resized = mask_pil.resize(
        (original_shape[1], original_shape[0]), 
        Image.Resampling.NEAREST
    )
    final_mask = np.array(mask_resized)
    
    # Light smoothing only if requested
    if apply_smoothing:
        pil_mask = Image.fromarray(final_mask)
        pil_mask = pil_mask.filter(ImageFilter.MedianFilter(3))  # Light median filter
        final_mask = np.array(pil_mask)
    
    return final_mask

def create_overlay(original_image, mask, alpha=0.6):
    """Create overlay visualization like your local version"""
    if isinstance(original_image, np.ndarray):
        original_pil = Image.fromarray(original_image.astype(np.uint8))
    else:
        original_pil = original_image
    
    original_rgba = original_pil.convert('RGBA')
    
    # Use semi-transparent red for overlay
    red_overlay = Image.new('RGBA', original_rgba.size, (255, 50, 50, int(255 * alpha)))
    
    mask_binary = mask > 0
    mask_pil = Image.fromarray((mask_binary * 255).astype(np.uint8)).convert('L')
    
    result = Image.composite(red_overlay, original_rgba, mask_pil)
    return result.convert('RGB')

# Main app interface
uploaded_file = st.file_uploader(
    "📤 Upload Satellite Image", 
    type=["jpg", "jpeg", "png", "tiff", "bmp"],
    help="Upload satellite imagery for oil spill analysis"
)

if uploaded_file is not None:
    try:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🛰️ Original Image")
            st.image(image, use_container_width=True)
            st.caption(f"Dimensions: {image.size}")
        
        with st.spinner("🔄 Analyzing image for oil spills..."):
            # Preprocess
            image_tensor, original_array, original_shape = preprocess_for_model(
                image, target_size=(target_size, target_size)
            )
            
            if model is not None:
                try:
                    with torch.no_grad():
                        prediction = model(image_tensor)
                    st.success("✅ AI Model Analysis Complete")
                    
                    # Simple processing like local version
                    final_mask = process_model_output(prediction, original_shape, confidence_threshold)
                    
                except Exception as e:
                    st.error(f"❌ Model inference failed: {e}")
                    # Fallback to empty mask
                    final_mask = np.zeros(original_shape[:2], dtype=np.uint8)
            else:
                st.error("❌ Model not available")
                final_mask = np.zeros(original_shape[:2], dtype=np.uint8)
            
            # Display results - MATCHING YOUR LOCAL LAYOUT
            with col2:
                st.subheader("Processed Image")
                mask_display = Image.fromarray(final_mask)
                st.image(mask_display, use_container_width=True, clamp=True)
                st.caption("Oil Spill Detection Mask")
            
            with col3:
                st.subheader("Oil Spill Overlay")
                overlay_result = create_overlay(original_array, final_mask)
                st.image(overlay_result, use_container_width=True)
                st.caption("Red areas show detected oil spills")
            
            # Simple analysis
            spill_pixels = np.sum(final_mask > 0)
            total_pixels = final_mask.size
            coverage_percent = (spill_pixels / total_pixels) * 100
            
            st.subheader("📊 Detection Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Spill Coverage", f"{coverage_percent:.2f}%")
            with col2:
                st.metric("Affected Pixels", f"{spill_pixels:,}")
            with col3:
                if spill_pixels > 0:
                    st.error("🚨 Oil Spill Detected")
                else:
                    st.success("✅ No Spill Detected")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

else:
    st.info("👆 Upload a satellite image to begin analysis")

# Simple status
if model is not None:
    st.sidebar.success("✅ Model Ready")
else:
    st.sidebar.error("❌ Model Not Available")

# Quick tips
st.sidebar.header("💡 Quick Tips")
st.sidebar.markdown("""
- Start with **confidence 0.5**
- Adjust up for **fewer detections**
- Adjust down for **more detections**
- Enable **smoothing** for cleaner edges
""")
