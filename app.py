import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import io
import collections
import os

# Set page config
st.set_page_config(page_title="Oil Spill Detection", layout="wide")

st.title("🛢️ Oil Spill Detection")
st.write("Upload satellite imagery to detect oil spills")

# First, let's find the model file
st.sidebar.header("🔍 Model File Detection")

def find_model_files():
    """Find all potential model files in current directory"""
    all_files = os.listdir('.')
    model_files = []
    
    for file in all_files:
        if any(file.endswith(ext) for ext in ['.pth', '.pt', '.pkl', '.h5', '.keras', '.onnx']):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            model_files.append((file, size_mb))
    
    return model_files

model_files = find_model_files()

if model_files:
    st.sidebar.success(f"✅ Found {len(model_files)} model file(s):")
    for file, size in model_files:
        st.sidebar.write(f"📦 {file} ({size:.1f} MB)")
else:
    st.sidebar.error("❌ No model files found!")
    st.sidebar.info("""
    **Please ensure:**
    1. Your model file is in the same directory as app.py
    2. File has extension: .pth, .pt, .pkl, or .h5
    3. File is committed to Git (if deploying)
    """)

# Model architecture
class OilSpillModel(nn.Module):
    def __init__(self, num_classes=1):
        super(OilSpillModel, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return torch.sigmoid(x)

@st.cache_resource
def load_model():
    """Try to load any model file we find"""
    model_files = find_model_files()
    
    if not model_files:
        st.sidebar.error("❌ No model files found to load!")
        return None
    
    # Try each model file
    for model_file, size in model_files:
        try:
            st.sidebar.info(f"🔄 Trying to load: {model_file}")
            checkpoint = torch.load(model_file, map_location='cpu')
            
            if isinstance(checkpoint, collections.OrderedDict):
                model = OilSpillModel(num_classes=1)
                model.load_state_dict(checkpoint, strict=False)
                model.eval()
                st.sidebar.success(f"✅ Loaded {model_file} successfully!")
                return model
            elif isinstance(checkpoint, torch.nn.Module):
                checkpoint.eval()
                st.sidebar.success(f"✅ Loaded full model from {model_file}!")
                return checkpoint
                
        except Exception as e:
            st.sidebar.warning(f"⚠️ Failed to load {model_file}: {str(e)[:100]}...")
            continue
    
    st.sidebar.error("❌ All model files failed to load!")
    return None

# Load model
model = load_model()

# Demo mode if no model is found
if model is None:
    st.warning("""
    ⚠️ **No model loaded - Running in DEMO MODE**
    
    The app will show sample outputs but won't perform real detection.
    
    **To fix this:**
    1. **Ensure your model file is in the repository**
    2. **Common model file names:**
       - `model.pth`, `model.pt`, `unet_model.pth`
       - `oil_spill_model.pth`, `segmentation_model.pth`
    3. **Check your repository structure:**
    ```
    your-repo/
    ├── app.py
    ├── requirements.txt
    ├── your-model-file.pth  # ← This should exist!
    └── other files...
    ```
    """)
    
    # Create a dummy model for demo
    class DemoModel:
        def __init__(self):
            self.is_demo = True
        
        def __call__(self, x):
            # Return random "detections" for demo
            batch, channels, height, width = x.shape
            fake_output = torch.rand(1, 1, height, width) * 0.1  # Low confidence
            return torch.sigmoid(fake_output)
    
    model = DemoModel()

# Simple preprocessing
def preprocess_image(image, size=512):
    original_size = image.size
    if image.mode != 'RGB': 
        image = image.convert('RGB')
    image_resized = image.resize((size, size))
    
    img_array = np.array(image_resized) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    return img_tensor, original_size

# Main application
st.header("📡 Upload Satellite Imagery")

uploaded_file = st.file_uploader(
    "Choose satellite image", 
    type=['png', 'jpg', 'jpeg', 'tiff', 'bmp']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    original_size = image.size
    
    # Display layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🛰️ Original Image")
        st.image(image, use_container_width=True)
        st.write(f"Size: {original_size}")
    
    # Check if we're in demo mode
    is_demo = hasattr(model, 'is_demo') and model.is_demo
    
    if is_demo:
        st.warning("🔸 **DEMO MODE**: Showing sample detection (not real AI)")
    
    # Process and predict
    with st.spinner("🔍 Analyzing for oil spills..." if not is_demo else "🔸 Demo mode..."):
        try:
            # Preprocess
            input_tensor, original_size = preprocess_image(image, size=512)
            
            if not is_demo:
                # Move to model device for real model
                device = next(model.parameters()).device
                input_tensor = input_tensor.to(device)
            
            # Prediction
            with torch.no_grad():
                output = model(input_tensor)
            
            # Handle output
            if isinstance(output, (list, tuple)):
                output = output[0]
            
            # Get probability map
            prob_map = output.squeeze().cpu().numpy()
            
            # For demo mode, add some "fake" detections
            if is_demo:
                # Create some artificial "oil spill" patterns for demo
                h, w = prob_map.shape
                y, x = np.ogrid[:h, :w]
                center_y, center_x = h // 2, w // 2
                mask = ((x - center_x)**2 + (y - center_y)**2 <= min(h, w)**2 // 16)
                prob_map[mask] = np.random.uniform(0.3, 0.8, size=mask.sum())
            
            # Use auto threshold
            prob_255 = (prob_map * 255).astype(np.uint8)
            if len(np.unique(prob_255)) > 1:
                try:
                    threshold, _ = cv2.threshold(prob_255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    confidence_threshold = threshold / 255.0
                except:
                    confidence_threshold = 0.5
            else:
                confidence_threshold = 0.5
            
            # Create mask
            binary_mask = (prob_map > confidence_threshold).astype(np.uint8) * 255
            binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
            
            # Create overlay
            original_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            colored_mask = np.zeros_like(original_cv)
            colored_mask[binary_mask_resized > 0] = [0, 0, 255]  # Red for oil
            blended = cv2.addWeighted(original_cv, 0.7, colored_mask, 0.3, 0)
            blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            
            # Display results
            with col2:
                st.subheader("🎭 Detection Mask")
                st.image(binary_mask_resized, use_container_width=True)
                st.write(f"Threshold: {confidence_threshold:.3f}")
                if is_demo:
                    st.caption("🔸 Demo detection pattern")
            
            with col3:
                st.subheader("🛢️ Oil Spill Overlay")
                st.image(blended_rgb, use_container_width=True)
                if is_demo:
                    st.caption("🔸 Demo overlay")
            
            # Statistics
            st.subheader("📊 Detection Analysis")
            
            total_pixels = binary_mask_resized.size
            oil_pixels = np.sum(binary_mask_resized > 0)
            oil_percentage = (oil_pixels / total_pixels) * 100
            
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            
            with col_stats1:
                st.metric("Oil Pixels", f"{oil_pixels:,}")
            with col_stats2:
                st.metric("Coverage", f"{oil_percentage:.4f}%")
            with col_stats3:
                st.metric("Max Confidence", f"{np.max(prob_map):.3f}")
            
            # Alert system
            st.subheader("🚨 Detection Alert")
            if oil_pixels > 0:
                if oil_percentage > 1:
                    st.error(f"🚨 OIL SPILL DETECTED! {oil_percentage:.4f}% coverage")
                else:
                    st.warning(f"⚠️ Potential oil sheen: {oil_percentage:.4f}% coverage")
            else:
                st.success("✅ No oil spills detected")
            
            if is_demo:
                st.info("🔸 This is a DEMO. Upload your model file for real detection.")
            
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")

# File upload option for model
with st.expander("📤 Upload Model File (Alternative)"):
    st.info("If your model file isn't in the repository, you can upload it directly:")
    
    uploaded_model = st.file_uploader(
        "Upload your model file", 
        type=['pth', 'pt', 'pkl'],
        help="Upload .pth, .pt, or .pkl model files"
    )
    
    if uploaded_model is not None:
        try:
            # Save uploaded file
            with open("uploaded_model.pth", "wb") as f:
                f.write(uploaded_model.getvalue())
            
            st.success("✅ Model file uploaded! Refresh the app to use it.")
            st.info("The app should automatically detect and load 'uploaded_model.pth' after refresh.")
            
        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")

# Troubleshooting guide
with st.expander("🔧 Setup Guide"):
    st.markdown("""
    **To get this working with your model:**
    
    1. **Ensure your model file is in the repository:**
       ```bash
       # Check what files exist
       ls -la
       
       # If your file has a different name, rename it:
       mv your_actual_model_file.pth oil_spill_model_deploy.pth
       ```
    
    2. **Common model file locations:**
       - Same directory as `app.py`
       - In a `models/` folder
       - Named: `model.pth`, `unet.pth`, `best_model.pth`
    
    3. **For large files (>100MB):**
       ```bash
       # Use Git LFS for large model files
       git lfs install
       git lfs track "*.pth"
       git add .gitattributes
       git add your_model.pth
       ```
    
    4. **Check your repository structure:**
       ```
       your-repo/
       ├── app.py
       ├── requirements.txt
       ├── oil_spill_model_deploy.pth  # ← Your model file
       └── other files...
       ```
    """)
