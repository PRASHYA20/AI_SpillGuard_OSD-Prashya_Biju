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
st.set_page_config(page_title="Oil Spill Detection - Overlay Fix", layout="wide")

st.title("🔧 Oil Spill Detection - Fixing Wrong Overlays")
st.write("Let's fix the incorrect detection areas")

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
    try:
        checkpoint = torch.load('oil_spill_model_deploy.pth', map_location='cpu')
        if isinstance(checkpoint, collections.OrderedDict):
            model = OilSpillModel(num_classes=1)
            model.load_state_dict(checkpoint, strict=False)
            model.eval()
            st.sidebar.success("✅ Model loaded")
            return model
    except Exception as e:
        st.sidebar.error(f"❌ Loading failed: {str(e)}")
        return None

model = load_model()

# Enhanced preprocessing with debugging
def preprocess_and_debug(image, size=512, method='standard'):
    """Preprocess with detailed debugging"""
    original_size = image.size
    
    st.sidebar.subheader("🔍 Preprocessing Debug")
    st.sidebar.write(f"Original size: {original_size}")
    st.sidebar.write(f"Target size: {size}x{size}")
    st.sidebar.write(f"Method: {method}")
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        st.sidebar.write(f"Converted from {image.mode} to RGB")
        image = image.convert('RGB')
    
    # Resize
    image_resized = image.resize((size, size))
    img_array = np.array(image_resized)
    
    st.sidebar.write(f"Resized array shape: {img_array.shape}")
    st.sidebar.write(f"Value range: {np.min(img_array)} to {np.max(img_array)}")
    
    # Different preprocessing strategies
    if method == 'no_normalize':
        img_array = img_array.astype(np.float32) / 255.0
        st.sidebar.write("Using: Simple 0-1 normalization")
        
    elif method == 'simple_normalize':
        img_array = img_array.astype(np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5
        st.sidebar.write("Using: Simple mean/std normalization")
        
    elif method == 'opencv_style':
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_array = img_array.astype(np.float32) / 255.0
        mean = np.array([0.406, 0.456, 0.485])
        std = np.array([0.225, 0.224, 0.229])
        img_array = (img_array - mean) / std
        st.sidebar.write("Using: BGR + ImageNet stats")
        
    else:  # standard
        img_array = img_array.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        st.sidebar.write("Using: RGB + ImageNet stats")
    
    st.sidebar.write(f"Normalized range: {np.min(img_array):.3f} to {np.max(img_array):.3f}")
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    
    st.sidebar.write(f"Tensor shape: {img_tensor.shape}")
    
    return img_tensor, original_size

def create_better_overlay(original_image, mask, prob_map=None, overlay_type='standard'):
    """Create better overlays with different visualization options"""
    original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    
    if overlay_type == 'standard':
        # Standard red overlay
        colored_mask = np.zeros_like(original_cv)
        colored_mask[mask > 0] = [0, 0, 255]  # Red
        blended = cv2.addWeighted(original_cv, 0.7, colored_mask, 0.3, 0)
        
    elif overlay_type == 'transparent_red':
        # More transparent red
        colored_mask = np.zeros_like(original_cv)
        colored_mask[mask > 0] = [0, 0, 255]
        blended = cv2.addWeighted(original_cv, 0.8, colored_mask, 0.2, 0)
        
    elif overlay_type == 'yellow_highlight':
        # Yellow highlight
        colored_mask = np.zeros_like(original_cv)
        colored_mask[mask > 0] = [0, 255, 255]  # Yellow
        blended = cv2.addWeighted(original_cv, 0.6, colored_mask, 0.4, 0)
        
    elif overlay_type == 'confidence_based' and prob_map is not None:
        # Color based on confidence
        colored_mask = np.zeros_like(original_cv)
        mask_resized = cv2.resize(mask, (prob_map.shape[1], prob_map.shape[0]))
        
        # High confidence = red, medium = orange, low = yellow
        high_conf = (prob_map > 0.7) & (mask_resized > 0)
        med_conf = (prob_map > 0.3) & (prob_map <= 0.7) & (mask_resized > 0)
        low_conf = (prob_map <= 0.3) & (mask_resized > 0)
        
        colored_mask[high_conf] = [0, 0, 255]    # Red
        colored_mask[med_conf] = [0, 165, 255]   # Orange
        colored_mask[low_conf] = [0, 255, 255]   # Yellow
        
        colored_mask = cv2.resize(colored_mask, (original_cv.shape[1], original_cv.shape[0]))
        blended = cv2.addWeighted(original_cv, 0.7, colored_mask, 0.3, 0)
        
    elif overlay_type == 'outline_only':
        # Just outline the detected areas
        blended = original_cv.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 0, 255), 3)  # Red outline
        
    else:
        blended = original_cv
    
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    return blended_rgb

# Settings
st.sidebar.header("⚙️ Overlay Settings")

# Preprocessing options
preprocessing_method = st.sidebar.selectbox(
    "Preprocessing Method",
    ['standard', 'no_normalize', 'simple_normalize', 'opencv_style'],
    help="Try different preprocessing methods"
)

input_size = st.sidebar.selectbox("Input Size", [256, 384, 512, 768], index=2)

# Threshold settings
st.sidebar.subheader("Threshold Settings")
threshold_type = st.sidebar.radio("Threshold Type", ['auto_otsu', 'manual', 'percentile'])

if threshold_type == 'manual':
    confidence_threshold = st.sidebar.slider("Manual Threshold", 0.001, 0.999, 0.5, 0.01)
elif threshold_type == 'percentile':
    percentile = st.sidebar.slider("Percentile", 50, 99, 95)
else:
    confidence_threshold = None

# Overlay settings
st.sidebar.subheader("Overlay Style")
overlay_style = st.sidebar.selectbox(
    "Overlay Type",
    ['standard', 'transparent_red', 'yellow_highlight', 'confidence_based', 'outline_only'],
    help="Different ways to visualize detections"
)

# Post-processing
st.sidebar.subheader("Post-processing")
enable_cleaning = st.sidebar.checkbox("Clean Small Objects", value=True)
min_object_size = st.sidebar.slider("Min Object Size", 10, 5000, 100, 10)
invert_mask = st.sidebar.checkbox("Invert Mask", value=False, help="Try this if detection is inverted")

# Main application
if model is not None:
    st.header("📡 Upload Image with Wrong Detection")
    
    uploaded_file = st.file_uploader("Choose image where overlay is wrong", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display original
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🛰️ Original Image")
            st.image(image, use_container_width=True)
            st.write(f"Size: {image.size}")
        
        # Process and predict
        with st.spinner("🔍 Analyzing detection..."):
            try:
                # Preprocess
                input_tensor, original_size = preprocess_and_debug(
                    image, 
                    size=input_size, 
                    method=preprocessing_method
                )
                
                # Move to model device
                device = next(model.parameters()).device
                input_tensor = input_tensor.to(device)
                
                # Prediction
                with torch.no_grad():
                    output = model(input_tensor)
                
                if isinstance(output, (list, tuple)):
                    output = output[0]
                
                # Get probability map
                prob_map = output.squeeze().cpu().numpy()
                
                # Calculate threshold
                if threshold_type == 'auto_otsu':
                    prob_255 = (prob_map * 255).astype(np.uint8)
                    if len(np.unique(prob_255)) > 1:
                        threshold, _ = cv2.threshold(prob_255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        confidence_threshold = threshold / 255.0
                    else:
                        confidence_threshold = 0.5
                elif threshold_type == 'percentile':
                    confidence_threshold = np.percentile(prob_map, percentile) / 100.0
                
                st.info(f"**Using threshold: {confidence_threshold:.4f}**")
                
                # Create mask
                binary_mask = (prob_map > confidence_threshold).astype(np.uint8) * 255
                
                # Invert mask if needed
                if invert_mask:
                    binary_mask = 255 - binary_mask
                    st.warning("🔄 Mask inverted - try this if detection was backwards")
                
                # Resize to original
                binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
                
                # Clean small objects
                if enable_cleaning:
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask_resized, connectivity=8)
                    cleaned_mask = np.zeros_like(binary_mask_resized)
                    for i in range(1, num_labels):
                        if stats[i, cv2.CC_STAT_AREA] >= min_object_size:
                            cleaned_mask[labels == i] = 255
                    binary_mask_resized = cleaned_mask
                
                # Create overlays
                overlay_standard = create_better_overlay(image, binary_mask_resized, prob_map, 'standard')
                overlay_selected = create_better_overlay(image, binary_mask_resized, prob_map, overlay_style)
                
                # Display results
                with col2:
                    st.subheader("🎭 Detection Mask")
                    st.image(binary_mask_resized, use_container_width=True)
                    st.write(f"Threshold: {confidence_threshold:.4f}")
                    
                    # Show mask stats
                    oil_pixels = np.sum(binary_mask_resized > 0)
                    total_pixels = binary_mask_resized.size
                    coverage = (oil_pixels / total_pixels) * 100
                    st.write(f"Coverage: {coverage:.4f}%")
                    st.write(f"Oil pixels: {oil_pixels:,}")
                
                with col3:
                    st.subheader(f"🛢️ {overlay_style.replace('_', ' ').title()}") 
                    st.image(overlay_selected, use_container_width=True)
                    st.write(f"Style: {overlay_style}")
                
                # Detailed analysis
                st.subheader("📊 Model Output Analysis")
                
                # Confidence statistics
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    st.write("**Confidence Statistics:**")
                    st.write(f"- Min: {np.min(prob_map):.6f}")
                    st.write(f"- Max: {np.max(prob_map):.6f}")
                    st.write(f"- Mean: {np.mean(prob_map):.6f}")
                    st.write(f"- Std: {np.std(prob_map):.6f}")
                    st.write(f"- Median: {np.median(prob_map):.6f}")
                
                with analysis_col2:
                    st.write("**Detection Quality:**")
                    if np.max(prob_map) < 0.01:
                        st.error("❌ Model very uncertain")
                    elif np.mean(prob_map) > 0.9:
                        st.warning("⚠️ Model overconfident")
                    elif coverage > 50:
                        st.warning("⚠️ Too much area detected")
                    elif coverage < 0.01:
                        st.info("ℹ️ Very little detection")
                    else:
                        st.success("✅ Output looks reasonable")
                
                # Test multiple thresholds
                st.subheader("🔍 Test Multiple Thresholds")
                test_thresholds = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
                test_cols = st.columns(7)
                
                for col, thresh in zip(test_cols, test_thresholds):
                    with col:
                        test_mask = (prob_map > thresh).astype(np.uint8) * 255
                        test_mask_resized = cv2.resize(test_mask, original_size, interpolation=cv2.INTER_NEAREST)
                        
                        test_pixels = np.sum(test_mask_resized > 0)
                        test_coverage = (test_pixels / total_pixels) * 100
                        
                        st.image(test_mask_resized, use_container_width=True, caption=f"Thresh: {thresh}")
                        st.write(f"{test_coverage:.2f}%")
                
                # Show probability heatmap
                st.subheader("🎨 Probability Heatmap")
                prob_display = (prob_map * 255).astype(np.uint8)
                prob_colored = cv2.applyColorMap(prob_display, cv2.COLORMAP_JET)
                prob_resized = cv2.resize(prob_colored, original_size)
                st.image(prob_resized, use_container_width=True, caption="Confidence Heatmap (Red = High, Blue = Low)")
                
            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")

else:
    st.error("Model failed to load")

# Quick fixes for common overlay issues
with st.expander("🔧 Common Overlay Issues & Fixes"):
    st.markdown("""
    **If the overlay is wrong, try these fixes:**
    
    ### 🎯 **Threshold Issues:**
    - **Too much detection?** → Increase threshold (0.7-0.9)
    - **Too little detection?** → Decrease threshold (0.1-0.3)
    - **Try auto_otsu** for automatic threshold calculation
    
    ### 🔄 **Inverted Detection:**
    - **Enable 'Invert Mask'** if oil is detected as background
    - This happens if the model was trained with opposite labels
    
    ### 🎨 **Overlay Style:**
    - **'outline_only'** - Shows only contours (good for precise areas)
    - **'confidence_based'** - Colors based on model confidence
    - **'transparent_red'** - Less intrusive overlay
    
    ### ⚙️ **Preprocessing:**
    - Try **'no_normalize'** if model expects simple 0-1 inputs
    - Try **'opencv_style'** if model was trained with BGR images
    
    ### 🧹 **Post-processing:**
    - **Enable cleaning** to remove small noisy detections
    - Adjust **min object size** based on expected spill size
    """)
