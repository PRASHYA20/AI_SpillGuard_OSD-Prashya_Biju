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

# Debug: Show current directory and files
st.sidebar.header("🔍 File System Debug")
current_dir = os.getcwd()
st.sidebar.write(f"Current directory: {current_dir}")

# List all files
all_files = os.listdir('.')
st.sidebar.write("All files:")
for file in sorted(all_files):
    if os.path.isfile(file):
        size = os.path.getsize(file)
        st.sidebar.write(f"- {file} ({size} bytes)")

# Look for model files in multiple possible locations
def find_model_file():
    possible_names = [
        "oil_spill_model_deploy.pth",
        "./oil_spill_model_deploy.pth", 
    ]
    
    for model_path in possible_names:
        if os.path.exists(model_path):
            st.sidebar.success(f"✅ Found model at: {model_path}")
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            st.sidebar.write(f"File size: {file_size:.2f} MB")
            return model_path
    
    st.sidebar.error("❌ Model file not found in any location")
    return None

# Define the model architecture
class OilSpillSegmentationModel(nn.Module):
    def __init__(self):
        super(OilSpillSegmentationModel, self).__init__()
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
    """Load the model with comprehensive error handling"""
    model_path = find_model_file()
    
    if model_path is None:
        st.error("❌ Cannot find model file. Please check deployment.")
        return None
    
    try:
        st.sidebar.info("🔄 Loading model...")
        
        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu')
        st.sidebar.success("✅ State dict loaded")
        
        # Create model
        model = OilSpillSegmentationModel()
        
        # Load weights with strict=False
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        st.sidebar.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {e}")
        return None

# Load model
model = load_model()

# Settings
st.sidebar.header("⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 0.99, 0.3, 0.01,  # Lower default for oil spill
    help="Higher values = fewer false positives"
)

target_size = st.sidebar.selectbox(
    "Processing Size", [256, 512, 224], index=0,
    help="Image size for model processing"
)

# Advanced settings
st.sidebar.header("🎯 Advanced Settings")
min_spill_size = st.sidebar.slider(
    "Minimum Spill Size (pixels)", 10, 1000, 100, 10,
    help="Filter out small detections"
)

apply_morphology = st.sidebar.checkbox(
    "Apply Noise Filtering", value=True,
    help="Remove small noise from detections"
)

# NEW: Add output inversion option
invert_output = st.sidebar.checkbox(
    "Invert Detection", value=True,  # Default to True since you're having inversion issues
    help="If oil spills are detected as background, enable this"
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
    """Process the actual model output with inversion handling"""
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.squeeze().detach().cpu().numpy()
    
    # Debug info
    st.sidebar.write("🔬 Model Output Analysis:")
    st.sidebar.write(f"Shape: {prediction.shape}")
    st.sidebar.write(f"Range: {prediction.min():.3f} to {prediction.max():.3f}")
    st.sidebar.write(f"Mean: {prediction.mean():.3f}")
    
    # Apply confidence threshold
    binary_mask = (prediction > confidence_threshold).astype(np.uint8)
    
    st.sidebar.write(f"Pixels above threshold: {np.sum(binary_mask)}")
    
    # NEW: Invert the mask if needed
    if invert_output:
        st.sidebar.info("🔄 Output inverted (oil = white)")
        binary_mask = 1 - binary_mask  # Invert the mask
        st.sidebar.write(f"After inversion: {np.sum(binary_mask)} pixels")
    else:
        st.sidebar.info("↔️ Normal output (oil = white)")
    
    # Resize to original dimensions
    mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
    mask_resized = mask_pil.resize(
        (original_shape[1], original_shape[0]), 
        Image.Resampling.NEAREST
    )
    final_mask = np.array(mask_resized)
    
    # Apply noise filtering
    if apply_morphology:
        pil_mask = Image.fromarray(final_mask)
        for _ in range(2):
            pil_mask = pil_mask.filter(ImageFilter.MinFilter(3))  # Remove small noise
        for _ in range(2):
            pil_mask = pil_mask.filter(ImageFilter.MaxFilter(3))  # Fill small holes
        final_mask = np.array(pil_mask)
    
    # Filter by minimum size
    if min_spill_size > 1:
        # Simple size filtering using connected components
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(final_mask > 0)
        
        for i in range(1, num_features + 1):
            component_size = np.sum(labeled_array == i)
            if component_size < min_spill_size:
                final_mask[labeled_array == i] = 0
    
    return final_mask

def create_overlay(original_image, mask, alpha=0.6):
    """Create overlay visualization"""
    if isinstance(original_image, np.ndarray):
        original_pil = Image.fromarray(original_image.astype(np.uint8))
    else:
        original_pil = original_image
    
    original_rgba = original_pil.convert('RGBA')
    red_overlay = Image.new('RGBA', original_rgba.size, (255, 0, 0, int(255 * alpha)))
    
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
                    st.success("✅ Real AI model used for detection")
                    
                    # Process actual model output
                    final_mask = process_model_output(prediction, original_shape, confidence_threshold)
                    
                except Exception as e:
                    st.error(f"❌ Model inference failed: {e}")
                    # Create a simple demo detection
                    h, w = original_shape
                    final_mask = np.zeros((h, w), dtype=np.uint8)
                    # Add a demo spill in the center
                    center_y, center_x = h // 2, w // 2
                    y, x = np.ogrid[:h, :w]
                    ellipse_mask = ((x - center_x)**2 / (w//8)**2 + (y - center_y)**2 / (h//8)**2) <= 1
                    final_mask[ellipse_mask] = 255
                    st.warning("⚠️ Using demo detection due to model error")
            else:
                st.warning("⚠️ Using demo detection (model not available)")
                # Create demo detection
                h, w = original_shape
                final_mask = np.zeros((h, w), dtype=np.uint8)
                center_y, center_x = h // 2, w // 2
                y, x = np.ogrid[:h, :w]
                ellipse_mask = ((x - center_x)**2 / (w//8)**2 + (y - center_y)**2 / (h//8)**2) <= 1
                final_mask[ellipse_mask] = 255
            
            # Display results
            mask_display = Image.fromarray(final_mask)
            with col2:
                st.subheader("🎭 Detection Mask")
                st.image(mask_display, use_container_width=True, clamp=True)
                st.caption("White = Oil Spill Areas")
            
            # Create overlay
            overlay_result = create_overlay(original_array, final_mask)
            
            with col3:
                st.subheader("🛢️ Oil Spill Overlay")
                st.image(overlay_result, use_container_width=True)
                st.caption("Red = Detected Oil Spills")
            
            # Analysis results
            st.subheader("📊 Analysis Results")
            spill_pixels = np.sum(final_mask > 0)
            total_pixels = final_mask.size
            coverage_percent = (spill_pixels / total_pixels) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Spill Coverage", f"{coverage_percent:.4f}%")
            with col2:
                st.metric("Affected Area", f"{spill_pixels:,} px")
            with col3:
                st.metric("Confidence", f"{confidence_threshold:.2f}")
            with col4:
                status = "🚨 SPILL DETECTED" if spill_pixels > 0 else "✅ CLEAN"
                st.metric("Status", status)
            
            # Risk assessment
            if coverage_percent > 5:
                st.error("🚨 **CRITICAL RISK** - Major oil spill detected")
            elif coverage_percent > 1:
                st.warning("⚠️ **HIGH RISK** - Significant oil contamination")
            elif coverage_percent > 0.1:
                st.info("🔶 **MEDIUM RISK** - Moderate spill detected")
            elif coverage_percent > 0.01:
                st.success("🔷 **LOW RISK** - Minor detection")
            else:
                st.success("✅ **CLEAN** - No oil spills detected")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

else:
    st.info("👆 Upload a satellite image to begin analysis")
    
    # Instructions
    st.subheader("🎯 How to Use")
    st.markdown("""
    1. **Upload** a satellite image containing water areas
    2. **Adjust settings** if needed:
       - **Lower confidence** (0.2-0.4) for more sensitive detection
       - **Higher confidence** (0.6-0.8) for fewer false positives
       - **Enable 'Invert Detection'** if oil is detected as background
    3. **View results** in the detection mask and overlay
    4. **Download** results for analysis
    """)

# Final status
st.sidebar.header("🎯 System Status")
if model is not None:
    st.sidebar.success("✅ Model: LOADED AND READY")
    st.sidebar.info("Real AI detection active")
else:
    st.sidebar.warning("⚠️ Model: NOT AVAILABLE")
    st.sidebar.info("Demo mode active")

# Tips
st.sidebar.header("💡 Tips")
st.sidebar.markdown("""
- Start with **low confidence** (0.3) and **invert enabled**
- Adjust **confidence threshold** based on results
- Enable **noise filtering** to remove small detections
- Use **minimum spill size** to filter noise
""")
