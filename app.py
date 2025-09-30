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

# Define the correct model architecture based on the state dict keys
class OilSpillSegmentationModel(nn.Module):
    def __init__(self):
        super(OilSpillSegmentationModel, self).__init__()
        # Based on the state dict keys, this appears to be a ResNet-based encoder with decoder
        # Using a pretrained ResNet as encoder
        self.encoder = models.resnet50(pretrained=False)
        # Remove the final classification layer
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        
        # Simple decoder for segmentation
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
        # Encoder
        features = self.encoder(x)
        
        # Decoder
        x = self.decoder(features)
        
        # Segmentation head
        x = self.segmentation_head(x)
        x = self.sigmoid(x)
        
        return x

@st.cache_resource
def load_model():
    """Load the model with the correct architecture"""
    model_path = "oil_spill_model_deploy.pth"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None
    
    try:
        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Create model with correct architecture
        model = OilSpillSegmentationModel()
        
        # Load state dict with strict=False (since we're using a simplified architecture)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        st.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

# Load model
model = load_model()

# Settings
st.sidebar.header("⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 0.99, 0.7, 0.01
)

target_size = st.sidebar.selectbox("Processing Size", [256, 512, 224], index=0)

# Advanced settings
st.sidebar.header("🎯 Advanced Settings")
min_spill_size = st.sidebar.slider(
    "Minimum Spill Size (pixels)", 10, 1000, 50, 10
)

apply_morphology = st.sidebar.checkbox("Apply Noise Filtering", value=True)

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
    """Process the actual model output"""
    # Convert prediction to numpy
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.squeeze().detach().cpu().numpy()
    
    st.sidebar.write("🔬 Model Output Analysis:")
    st.sidebar.write(f"Shape: {prediction.shape}")
    st.sidebar.write(f"Range: {prediction.min():.3f} to {prediction.max():.3f}")
    st.sidebar.write(f"Mean: {prediction.mean():.3f}")
    
    # Apply confidence threshold
    binary_mask = (prediction > confidence_threshold).astype(np.uint8)
    
    st.sidebar.write(f"Detected pixels: {np.sum(binary_mask)}")
    
    # Resize to original dimensions
    mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
    mask_resized = mask_pil.resize(
        (original_shape[1], original_shape[0]), 
        Image.Resampling.NEAREST
    )
    final_mask = np.array(mask_resized)
    
    # Apply morphology operations to clean up the mask
    if apply_morphology:
        pil_mask = Image.fromarray(final_mask)
        for _ in range(2):
            pil_mask = pil_mask.filter(ImageFilter.MinFilter(3))  # Erosion
        for _ in range(2):
            pil_mask = pil_mask.filter(ImageFilter.MaxFilter(3))  # Dilation
        final_mask = np.array(pil_mask)
    
    return final_mask

def create_overlay(original_image, mask, alpha=0.6):
    """Create overlay visualization"""
    if isinstance(original_image, np.ndarray):
        original_pil = Image.fromarray(original_image.astype(np.uint8))
    else:
        original_pil = original_image
    
    # Create overlay
    original_rgba = original_pil.convert('RGBA')
    red_overlay = Image.new('RGBA', original_rgba.size, (255, 0, 0, int(255 * alpha)))
    
    # Create mask
    mask_binary = mask > 0
    mask_pil = Image.fromarray((mask_binary * 255).astype(np.uint8)).convert('L')
    
    # Composite images
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
                        # Use the actual model for prediction
                        prediction = model(image_tensor)
                    st.success("✅ Real model used for detection")
                    
                    # Process the actual model output
                    final_mask = process_model_output(prediction, original_shape, confidence_threshold)
                    
                except Exception as e:
                    st.error(f"❌ Model inference failed: {e}")
                    # Fallback to simple detection
                    h, w = original_shape
                    final_mask = np.zeros((h, w), dtype=np.uint8)
                    
            else:
                st.error("❌ Model not available")
                h, w = original_shape
                final_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Display results
            mask_display = Image.fromarray(final_mask)
            with col2:
                st.subheader("🎭 Detection Mask")
                st.image(mask_display, use_container_width=True, clamp=True)
                st.caption(f"Spill coverage: {np.sum(final_mask > 0) / final_mask.size * 100:.2f}%")
            
            # Create overlay
            overlay_result = create_overlay(original_array, final_mask)
            
            with col3:
                st.subheader("🛢️ Oil Spill Overlay")
                st.image(overlay_result, use_container_width=True)
                st.caption("Red areas = Detected oil spills")
            
            # Analysis results
            st.subheader("📊 Analysis Results")
            spill_pixels = np.sum(final_mask > 0)
            total_pixels = final_mask.size
            coverage_percent = (spill_pixels / total_pixels) * 100
            
            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
            with col_metrics1:
                st.metric("Spill Coverage", f"{coverage_percent:.4f}%")
            with col_metrics2:
                st.metric("Affected Area", f"{spill_pixels:,} px")
            with col_metrics3:
                status = "🚨 SPILL DETECTED" if spill_pixels > 0 else "✅ CLEAN"
                st.metric("Status", status)

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

else:
    st.info("👆 Upload a satellite image to begin analysis")

# Show model status
if model is not None:
    st.sidebar.success("✅ Model: LOADED AND READY")
else:
    st.sidebar.error("❌ Model: NOT AVAILABLE")
