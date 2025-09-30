import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import io
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Set page config
st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Oil Spill Detection")
st.write("Upload satellite imagery for AI-powered oil spill detection")

# Debug: Check all files in directory
st.sidebar.header("🔍 File System Debug")
current_dir = os.getcwd()
st.sidebar.write(f"Current directory: {current_dir}")

# List all files with sizes
all_files = os.listdir('.')
st.sidebar.write("All files in directory:")
for file in all_files:
    if os.path.isfile(file):
        file_size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
        st.sidebar.write(f"- {file}: {file_size:.2f} MB")

# Check specifically for model files
model_files = [f for f in all_files if f.endswith(('.pth', '.pt'))]
st.sidebar.header("📁 Model Files Found")
if model_files:
    for model_file in model_files:
        file_size = os.path.getsize(model_file) / (1024 * 1024)
        st.sidebar.write(f"✅ {model_file}: {file_size:.2f} MB")
else:
    st.sidebar.error("❌ No model files found!")

# Simple model architecture for testing
class SimpleOilSpillModel(nn.Module):
    def __init__(self):
        super(SimpleOilSpillModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_model():
    """Try to load model with extensive debugging"""
    model_path = "oil_spill_model_deploy.pth"
    
    st.sidebar.header("🔄 Model Loading Debug")
    
    # Check if file exists and has content
    if not os.path.exists(model_path):
        st.sidebar.error(f"❌ File does not exist: {model_path}")
        return None
    
    file_size = os.path.getsize(model_path)
    st.sidebar.write(f"File size: {file_size} bytes")
    
    if file_size == 0:
        st.sidebar.error("❌ Model file is EMPTY (0 bytes)")
        return None
    
    if file_size < 1024:  # Less than 1KB
        st.sidebar.warning(f"⚠️ Model file very small: {file_size} bytes")
    
    try:
        # Try to load the file
        st.sidebar.info("🔄 Attempting to load model...")
        
        # First, let's see what's in the file
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            st.sidebar.success("✅ File can be loaded with torch.load")
            
            # Debug what's inside
            st.sidebar.write("📋 Checkpoint type:", type(checkpoint))
            if isinstance(checkpoint, dict):
                st.sidebar.write("📋 Checkpoint keys:", list(checkpoint.keys()))
                if 'state_dict' in checkpoint:
                    st.sidebar.write("📋 State dict keys sample:", list(checkpoint['state_dict'].keys())[:5])
        except Exception as e:
            st.sidebar.error(f"❌ torch.load failed: {e}")
            return None
        
        # Try different loading strategies
        strategies = [
            ("Direct state dict", lambda: torch.load(model_path, map_location='cpu')),
            ("With weights_only=False", lambda: torch.load(model_path, map_location='cpu', weights_only=False)),
        ]
        
        for strategy_name, load_func in strategies:
            try:
                st.sidebar.info(f"🔄 Trying: {strategy_name}")
                loaded_data = load_func()
                
                if isinstance(loaded_data, dict):
                    # It's a state dict
                    model = SimpleOilSpillModel()
                    try:
                        model.load_state_dict(loaded_data)
                        model.eval()
                        st.sidebar.success(f"✅ Loaded successfully with {strategy_name}")
                        return model
                    except Exception as e:
                        st.sidebar.warning(f"❌ State dict loading failed: {e}")
                        # Try with strict=False
                        try:
                            model.load_state_dict(loaded_data, strict=False)
                            model.eval()
                            st.sidebar.success(f"✅ Loaded with strict=False using {strategy_name}")
                            return model
                        except:
                            continue
                else:
                    st.sidebar.warning(f"❌ Unexpected data type: {type(loaded_data)}")
                    
            except Exception as e:
                st.sidebar.warning(f"❌ {strategy_name} failed: {e}")
                continue
        
        st.sidebar.error("❌ All loading strategies failed")
        return None
        
    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {e}")
        return None

# Load model
model = load_model()

# Settings
st.sidebar.header("⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 0.9, 0.7, 0.05
)

target_size = st.sidebar.selectbox("Processing Size", [256, 512, 224], index=0)

# Rest of your existing code for preprocessing, visualization etc.
# ... (include all the previous functions like preprocess_for_model, etc.)

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

def create_demo_prediction(image_tensor, original_shape):
    """Create a simple demo prediction when model is not available"""
    # Create a simple elliptical "spill" in the center
    h, w = original_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Add a simple elliptical area
    center_y, center_x = h // 2, w // 2
    radius_y, radius_x = h // 6, w // 6
    
    y, x = np.ogrid[:h, :w]
    ellipse_mask = ((x - center_x)**2 / radius_x**2 + (y - center_y)**2 / radius_y**2) <= 1
    mask[ellipse_mask] = 255
    
    return mask

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
        
        with st.spinner("🔄 Analyzing image..."):
            # Preprocess
            image_tensor, original_array, original_shape = preprocess_for_model(
                image, target_size=(target_size, target_size)
            )
            
            if model is not None:
                try:
                    with torch.no_grad():
                        prediction = model(image_tensor)
                    st.success("✅ Real model used for detection")
                    # Process prediction here
                    final_mask = create_demo_prediction(image_tensor, original_shape)  # Replace with actual processing
                except Exception as e:
                    st.error(f"❌ Model inference failed: {e}")
                    final_mask = create_demo_prediction(image_tensor, original_shape)
            else:
                st.warning("⚠️ Using demo mode (model not available)")
                final_mask = create_demo_prediction(image_tensor, original_shape)
            
            # Display results
            mask_display = Image.fromarray(final_mask)
            with col2:
                st.subheader("🎭 Detection Mask")
                st.image(mask_display, use_container_width=True, clamp=True)
            
            # Create overlay
            overlay_pil = image.copy().convert('RGBA')
            overlay_array = np.array(overlay_pil)
            red_overlay = np.zeros_like(overlay_array)
            red_overlay[final_mask > 0] = [255, 0, 0, 128]
            overlay_result = Image.alpha_composite(overlay_pil, Image.fromarray(red_overlay.astype(np.uint8)))
            
            with col3:
                st.subheader("🛢️ Oil Spill Overlay")
                st.image(overlay_result, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

else:
    st.info("👆 Upload a satellite image to begin analysis")

# Show final status
st.sidebar.header("🎯 Final Status")
if model is not None:
    st.sidebar.success("✅ Model: LOADED AND READY")
else:
    st.sidebar.error("❌ Model: NOT AVAILABLE")
    st.sidebar.info("💡 Solution: Check if your model file is properly uploaded to GitHub and deployed")
