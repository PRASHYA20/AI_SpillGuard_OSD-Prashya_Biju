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
st.set_page_config(page_title="Oil Spill Detection - File Fix", layout="wide")

st.title("🛢️ Oil Spill Detection - File Issues")
st.write("Let's find and fix the model file problem")

# First, let's debug the file situation
st.sidebar.header("🔍 File Debug Information")

def debug_files():
    """Show detailed file information"""
    st.sidebar.subheader("📁 Current Directory Files")
    
    current_dir = os.getcwd()
    st.sidebar.write(f"Working directory: `{current_dir}`")
    
    all_files = os.listdir('.')
    if not all_files:
        st.sidebar.error("❌ Directory is empty!")
        return []
    
    st.sidebar.write("All files:")
    for file in sorted(all_files):
        if os.path.isfile(file):
            size_kb = os.path.getsize(file) / 1024
            st.sidebar.write(f"📄 {file} ({size_kb:.1f} KB)")
        else:
            st.sidebar.write(f"📁 {file}/ (directory)")
    
    # Look for model files
    model_extensions = ['.pth', '.pt', '.pkl', '.h5', '.keras', '.onnx']
    model_files = [f for f in all_files if any(f.endswith(ext) for ext in model_extensions)]
    
    return model_files

# Run file debug
model_files = debug_files()

if model_files:
    st.sidebar.success(f"✅ Found {len(model_files)} model file(s)")
    for model_file in model_files:
        st.sidebar.write(f"🎯 {model_file}")
else:
    st.sidebar.error("❌ No model files found!")

# Create a demo mode that works without the model
st.sidebar.header("🎯 Demo Options")
demo_mode = st.sidebar.radio("Run Mode", ["Demo Pattern", "Upload Model File"])

# Model architecture (for when we have the file)
class OilSpillModel(nn.Module):
    def __init__(self, num_classes=1):
        super(OilSpillModel, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return torch.sigmoid(x)

def try_load_model():
    """Try to load any available model file"""
    if not model_files:
        return None
    
    for model_file in model_files:
        try:
            st.sidebar.info(f"🔄 Trying to load: {model_file}")
            checkpoint = torch.load(model_file, map_location='cpu')
            
            if isinstance(checkpoint, collections.OrderedDict):
                model = OilSpillModel(num_classes=1)
                model.load_state_dict(checkpoint, strict=False)
                model.eval()
                st.sidebar.success(f"✅ Loaded {model_file}")
                return model
            elif isinstance(checkpoint, torch.nn.Module):
                checkpoint.eval()
                st.sidebar.success(f"✅ Loaded full model: {model_file}")
                return checkpoint
                
        except Exception as e:
            st.sidebar.warning(f"⚠️ Failed to load {model_file}: {str(e)[:100]}")
            continue
    
    return None

# Try to load model
model = None
if demo_mode == "Upload Model File" and model_files:
    model = try_load_model()

# Demo model for testing overlays
class DemoModel:
    def __init__(self):
        self.is_demo = True
    
    def __call__(self, x):
        # Create realistic demo patterns
        batch, channels, height, width = x.shape
        
        # Create multiple "oil spill" patterns for demo
        output = torch.zeros(1, 1, height, width)
        
        # Pattern 1: Circular spill
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        center_y, center_x = height // 2, width // 2
        radius = min(height, width) // 4
        circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        output[0, 0, circle_mask] = torch.rand(circle_mask.sum()) * 0.5 + 0.5  # 0.5-1.0 confidence
        
        # Pattern 2: Random patches (like oil sheens)
        for _ in range(5):
            patch_y = torch.randint(0, height-20, (1,))
            patch_x = torch.randint(0, width-20, (1,))
            patch_size = torch.randint(10, 30, (1,))
            patch_mask = (x >= patch_x) & (x < patch_x + patch_size) & (y >= patch_y) & (y < patch_y + patch_size)
            output[0, 0, patch_mask] = torch.rand(1) * 0.3 + 0.2  # 0.2-0.5 confidence
        
        return torch.sigmoid(output)

if model is None:
    st.warning("""
    🔸 **RUNNING IN DEMO MODE**
    
    The app will show sample oil spill detections for testing the overlay system.
    To use your real model, ensure your model file is in the repository.
    """)
    model = DemoModel()

# Overlay settings
st.sidebar.header("🎨 Overlay Configuration")

overlay_style = st.sidebar.selectbox(
    "Overlay Style",
    ['red_fill', 'red_transparent', 'yellow_highlight', 'outline_only', 'confidence_heatmap'],
    help="Different visualization styles"
)

overlay_opacity = st.sidebar.slider("Overlay Opacity", 0.1, 1.0, 0.3, 0.1)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.01, 0.99, 0.5, 0.01)

def create_overlay(original_image, mask, prob_map=None, style='red_fill', opacity=0.3):
    """Create different overlay styles"""
    original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    
    if style == 'red_fill':
        # Solid red fill
        colored_mask = np.zeros_like(original_cv)
        colored_mask[mask > 0] = [0, 0, 255]
        blended = cv2.addWeighted(original_cv, 1 - opacity, colored_mask, opacity, 0)
        
    elif style == 'red_transparent':
        # More transparent red
        colored_mask = np.zeros_like(original_cv)
        colored_mask[mask > 0] = [0, 0, 255]
        blended = cv2.addWeighted(original_cv, 0.8, colored_mask, 0.2, 0)
        
    elif style == 'yellow_highlight':
        # Yellow highlight
        colored_mask = np.zeros_like(original_cv)
        colored_mask[mask > 0] = [0, 255, 255]  # Yellow in BGR
        blended = cv2.addWeighted(original_cv, 1 - opacity, colored_mask, opacity, 0)
        
    elif style == 'outline_only':
        # Just red outlines
        blended = original_cv.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 0, 255), 3)
        
    elif style == 'confidence_heatmap' and prob_map is not None:
        # Confidence-based coloring
        prob_resized = cv2.resize(prob_map, (original_cv.shape[1], original_cv.shape[0]))
        heatmap = cv2.applyColorMap((prob_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        mask_bool = mask > 0
        blended = original_cv.copy()
        blended[mask_bool] = heatmap[mask_bool]
        blended = cv2.addWeighted(original_cv, 0.5, blended, 0.5, 0)
        
    else:
        blended = original_cv
    
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    return blended_rgb

# Main application
st.header("📡 Test Overlay System")

uploaded_file = st.file_uploader("Upload any image to test overlays", type=['png', 'jpg', 'jpeg'])

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
        st.info("🔸 **Demo Mode**: Showing sample oil spill patterns")
    
    # Process image
    with st.spinner("Creating overlay..."):
        try:
            # Simple preprocessing for demo
            image_resized = image.resize((512, 512))
            img_array = np.array(image_resized) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = (img_array - mean) / std
            
            img_tensor = torch.from_numpy(img_array).float()
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # Get prediction (real or demo)
            with torch.no_grad():
                output = model(img_tensor)
            
            if isinstance(output, (list, tuple)):
                output = output[0]
            
            # Get probability map
            prob_map = output.squeeze().cpu().numpy()
            
            # Create mask
            binary_mask = (prob_map > confidence_threshold).astype(np.uint8) * 255
            binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
            
            # Create overlay
            overlay = create_overlay(
                image, 
                binary_mask_resized, 
                prob_map, 
                style=overlay_style,
                opacity=overlay_opacity
            )
            
            # Display results
            with col2:
                st.subheader("🎭 Detection Mask")
                st.image(binary_mask_resized, use_container_width=True)
                
                # Statistics
                oil_pixels = np.sum(binary_mask_resized > 0)
                total_pixels = binary_mask_resized.size
                coverage = (oil_pixels / total_pixels) * 100
                
                st.write(f"**Detection Stats:**")
                st.write(f"- Coverage: {coverage:.2f}%")
                st.write(f"- Oil Pixels: {oil_pixels:,}")
                st.write(f"- Threshold: {confidence_threshold}")
                
                if is_demo:
                    st.caption("🔸 Demo pattern - adjust threshold to see changes")
            
            with col3:
                st.subheader(f"🛢️ {overlay_style.replace('_', ' ').title()}")
                st.image(overlay, use_container_width=True)
                st.write(f"Style: `{overlay_style}`")
                st.write(f"Opacity: `{overlay_opacity}`")
            
            # Test different thresholds
            st.subheader("🔍 Test Different Thresholds")
            test_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
            test_cols = st.columns(5)
            
            for col, thresh in zip(test_cols, test_thresholds):
                with col:
                    test_mask = (prob_map > thresh).astype(np.uint8) * 255
                    test_mask_resized = cv2.resize(test_mask, original_size, interpolation=cv2.INTER_NEAREST)
                    
                    test_pixels = np.sum(test_mask_resized > 0)
                    test_coverage = (test_pixels / total_pixels) * 100
                    
                    st.image(test_mask_resized, use_container_width=True, caption=f"Thresh: {thresh}")
                    st.write(f"{test_coverage:.1f}% coverage")
            
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")

# File upload section for model
with st.expander("📤 Upload Your Model File"):
    st.markdown("""
    **If your model file isn't in the repository, upload it here:**
    """)
    
    uploaded_model = st.file_uploader(
        "Upload model file (.pth, .pt, .pkl)", 
        type=['pth', 'pt', 'pkl'],
        key="model_uploader"
    )
    
    if uploaded_model is not None:
        try:
            # Save the uploaded file
            model_filename = "uploaded_model.pth"
            with open(model_filename, "wb") as f:
                f.write(uploaded_model.getvalue())
            
            file_size = os.path.getsize(model_filename) / (1024 * 1024)
            st.success(f"✅ Model uploaded successfully! ({file_size:.1f} MB)")
            st.info("🔄 **Refresh the page** to load the uploaded model")
            
        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")

# Setup instructions
with st.expander("🔧 Setup Instructions"):
    st.markdown("""
    **To fix the model loading issue:**
    
    1. **Check your model file exists:**
    ```bash
    # In your terminal, run:
    ls -la *.pth *.pt *.pkl
    ```
    
    2. **If no model files found:**
       - Use the upload section above to upload your model
       - Or ensure your model file is committed to Git
    
    3. **Common model file names:**
       - `model.pth`, `model.pt`
       - `unet_model.pth`, `segmentation_model.pth` 
       - `oil_spill_model.pth`, `best_model.pth`
    
    4. **Repository structure should be:**
    ```
    your-repo/
    ├── app.py
    ├── requirements.txt
    ├── your-model-file.pth  # ← This must exist!
    └── other files...
    ```
    
    **Current directory files:** (from debug above)
    """)
    
    # Show files again
    files = os.listdir('.')
    for file in sorted(files):
        st.write(f"- `{file}`")

# Quick overlay guide
with st.expander("🎨 Overlay Style Guide"):
    st.markdown("""
    **Overlay Styles:**
    
    - **🔴 Red Fill**: Solid red areas for detected oil
    - **🔴 Red Transparent**: More subtle red overlay  
    - **🟡 Yellow Highlight**: Yellow areas (good for sheens)
    - **📐 Outline Only**: Red outlines around detected areas
    - **🌈 Confidence Heatmap**: Color shows model confidence
    
    **Tips:**
    - Use **lower opacity** (0.2-0.4) for subtle overlays
    - Use **outline only** for precise area visualization
    - **Adjust threshold** to control detection sensitivity
    """)
