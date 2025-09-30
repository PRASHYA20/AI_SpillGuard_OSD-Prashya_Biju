import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import io
import collections

# Set page config
st.set_page_config(page_title="Oil Spill Detection - Debug", layout="wide")

st.title("🔧 Oil Spill Detection Debug")
st.write("Let's figure out why detection isn't working properly")

# Model architecture (same as before)
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
            st.sidebar.success("✅ Model loaded with strict=False")
            return model
    except Exception as e:
        st.sidebar.error(f"❌ Loading failed: {str(e)}")
        return None

model = load_model()

def preprocess_simple(image, size=512):
    """Simple preprocessing"""
    original_size = image.size
    if image.mode != 'RGB': image = image.convert('RGB')
    image_resized = image.resize((size, size))
    
    img_array = np.array(image_resized) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    return img_tensor, original_size

if model is not None:
    st.header("📡 Upload Problematic Image")
    
    uploaded_file = st.file_uploader("Choose an image where detection fails", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            st.write(f"Size: {image.size}")
        
        # Process image
        input_tensor, original_size = preprocess_simple(image, size=512)
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, (list, tuple)): output = output[0]
            prob_map = output.squeeze().cpu().numpy()
        
        # Analyze model output in detail
        st.subheader("🔍 Detailed Model Output Analysis")
        
        # Basic statistics
        st.write("**Confidence Statistics:**")
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        
        with col_stats1:
            min_conf = np.min(prob_map)
            st.metric("Min", f"{min_conf:.6f}")
        with col_stats2:
            max_conf = np.max(prob_map)
            st.metric("Max", f"{max_conf:.6f}")
        with col_stats3:
            mean_conf = np.mean(prob_map)
            st.metric("Mean", f"{mean_conf:.6f}")
        with col_stats4:
            std_conf = np.std(prob_map)
            st.metric("Std", f"{std_conf:.6f}")
        
        # Percentile analysis
        st.write("**Percentile Analysis:**")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
        percentile_values = [np.percentile(prob_map, p) for p in percentiles]
        
        cols = st.columns(5)
        for i, (col, p, val) in enumerate(zip(cols, percentiles, percentile_values)):
            with col:
                st.write(f"{p}%: {val:.6f}")
        
        # Pixel count analysis
        st.write("**Pixel Count Analysis:**")
        thresholds = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        threshold_data = []
        
        for thresh in thresholds:
            pixels_above = np.sum(prob_map > thresh)
            percentage = (pixels_above / prob_map.size) * 100
            threshold_data.append((thresh, pixels_above, percentage))
        
        # Display threshold analysis
        st.write("**Threshold Analysis:**")
        analysis_cols = st.columns(3)
        for i, (col, (thresh, pixels, percent)) in enumerate(zip(analysis_cols, threshold_data)):
            with col:
                st.write(f"**>{thresh:.3f}**: {pixels:,} px ({percent:.4f}%)")
        
        # Visualize different thresholds
        st.subheader("🎯 Visual Threshold Testing")
        
        # Test very low thresholds to see if ANY detection appears
        st.write("**Testing Very Low Thresholds (0.001 to 0.1):**")
        low_thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
        low_cols = st.columns(5)
        
        for col, thresh in zip(low_cols, low_thresholds):
            with col:
                mask = (prob_map > thresh).astype(np.uint8) * 255
                mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
                
                # Create overlay
                original_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                colored = np.zeros_like(original_cv)
                colored[mask_resized > 0] = [0, 0, 255]
                blended = cv2.addWeighted(original_cv, 0.8, colored, 0.2, 0)
                blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
                
                pixels = np.sum(mask_resized > 0)
                coverage = (pixels / mask_resized.size) * 100
                
                st.image(blended_rgb, use_container_width=True, caption=f"Thresh: {thresh}")
                st.write(f"{coverage:.6f}% coverage")
        
        # Test normal thresholds
        st.write("**Testing Normal Thresholds (0.2 to 0.9):**")
        normal_thresholds = [0.2, 0.3, 0.5, 0.7, 0.9]
        normal_cols = st.columns(5)
        
        for col, thresh in zip(normal_cols, normal_thresholds):
            with col:
                mask = (prob_map > thresh).astype(np.uint8) * 255
                mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
                
                original_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                colored = np.zeros_like(original_cv)
                colored[mask_resized > 0] = [0, 0, 255]
                blended = cv2.addWeighted(original_cv, 0.8, colored, 0.2, 0)
                blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
                
                pixels = np.sum(mask_resized > 0)
                coverage = (pixels / mask_resized.size) * 100
                
                st.image(blended_rgb, use_container_width=True, caption=f"Thresh: {thresh}")
                st.write(f"{coverage:.6f}% coverage")
        
        # Show raw probability map
        st.subheader("📊 Raw Probability Visualization")
        
        # Create heatmap of probabilities
        prob_display = (prob_map * 255).astype(np.uint8)
        prob_colored = cv2.applyColorMap(prob_display, cv2.COLORMAP_JET)
        prob_resized = cv2.resize(prob_colored, original_size)
        
        col_prob1, col_prob2 = st.columns(2)
        with col_prob1:
            st.image(prob_resized, use_container_width=True, caption="Probability Heatmap")
        with col_prob2:
            # Show the most confident regions
            high_conf_mask = (prob_map > np.percentile(prob_map, 99)).astype(np.uint8) * 255
            high_conf_resized = cv2.resize(high_conf_mask, original_size)
            st.image(high_conf_resized, use_container_width=True, caption="Top 1% Most Confident Regions")
        
        # Diagnosis
        st.subheader("🩺 Diagnosis")
        
        if max_conf < 0.01:
            st.error("""
            **❌ CRITICAL ISSUE: Model is producing extremely low confidence values**
            
            **Possible causes:**
            1. **Wrong preprocessing** - Model expects different normalization
            2. **Architecture mismatch** - Model weights don't match our architecture
            3. **Trained on different data** - Model expects specific input characteristics
            4. **Model is broken** - Weights might be corrupted or incorrectly saved
            
            **Next steps:**
            - Try different preprocessing (BGR vs RGB, different normalization)
            - Check if model was trained on similar images
            - Verify the model architecture matches training
            """)
        elif max_conf < 0.1:
            st.warning("""
            **⚠️ Model is very uncertain**
            
            **Possible causes:**
            1. **Threshold too high** - Try thresholds below 0.1
            2. **Input mismatch** - Images differ from training data
            3. **Model needs calibration** - Outputs are not well-calibrated
            
            **Try:** Use thresholds between 0.01 and 0.1
            """)
        elif mean_conf > 0.8:
            st.warning("""
            **⚠️ Model is overconfident**
            
            **Possible causes:**
            1. **Overfitting** - Model memorized training data
            2. **Poor calibration** - Output probabilities don't reflect true confidence
            
            **Try:** Use higher thresholds (0.7-0.9)
            """)
        else:
            st.info("""
            **ℹ️ Model output looks reasonable**
            
            The issue might be:
            1. **Wrong threshold** - Adjust threshold up/down
            2. **Post-processing needed** - Clean up noisy detections
            3. **Training data mismatch** - Model trained on different types of oil spills
            
            **Try different thresholds and check if any look correct**
            """)

else:
    st.error("Model failed to load")

# Quick fixes to try
with st.expander("🚀 Quick Fixes to Try"):
    st.markdown("""
    **Based on the analysis above, try these:**
    
    **If max confidence < 0.01:**
    1. Try preprocessing with BGR instead of RGB
    2. Try no normalization (simple 0-1 scaling)
    3. Check if model expects different input size
    
    **If you see some detection but it's wrong:**
    1. Try threshold around the 95th percentile value
    2. Enable mask cleaning to remove small detections
    3. Try morphological operations to clean up the mask
    
    **If no detection at any threshold:**
    The model might not be working with your images. We may need to:
    - Get the exact model architecture from training
    - Check the training data characteristics
    - Retrain or fine-tune the model
    """)
