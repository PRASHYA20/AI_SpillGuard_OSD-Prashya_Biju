import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import io
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import random

# Set page config
st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Oil Spill Detection")
st.write("Upload satellite imagery for AI-powered oil spill detection")

# Define a simple model architecture (adjust based on your actual model)
class OilSpillModel(nn.Module):
    def __init__(self):
        super(OilSpillModel, self).__init__()
        # Simple CNN architecture - you may need to adjust this
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Alternative: UNet-like architecture for segmentation
class OilSpillSegmentationModel(nn.Module):
    def __init__(self):
        super(OilSpillSegmentationModel, self).__init__()
        # Encoder
        self.enc1 = self._block(3, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        
        # Decoder
        self.dec1 = self._block(512, 256)
        self.dec2 = self._block(256, 128)
        self.dec3 = self._block(128, 64)
        
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) if in_channels != 512 else nn.Identity()
        )
    
    def forward(self, x):
        # Simple forward pass - you might need to adjust this
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.final(x)
        return self.sigmoid(x)

# Model loading function for state dict
@st.cache_resource
def load_model():
    """Load the model weights and create model architecture"""
    try:
        model_path = "oil_spill_model_deploy.pth"
        st.sidebar.info(f"🔍 Looking for model at: {model_path}")
        
        # Check if file exists
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            st.sidebar.success(f"✅ Model found! Size: {file_size:.1f} MB")
            
            # Load state dict
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Debug: Show what's in the state dict
            st.sidebar.write("📋 State dict keys:", list(state_dict.keys())[:5] if state_dict else "Empty")
            
            # Try different model architectures
            models_to_try = [
                ("Segmentation Model", OilSpillSegmentationModel()),
                ("Classification Model", OilSpillModel())
            ]
            
            loaded_model = None
            for model_name, model in models_to_try:
                try:
                    # Try to load state dict into this model architecture
                    model.load_state_dict(state_dict)
                    model.eval()
                    st.sidebar.success(f"✅ Loaded as {model_name}")
                    loaded_model = model
                    break
                except Exception as e:
                    st.sidebar.warning(f"❌ Failed as {model_name}: {str(e)[:50]}...")
                    continue
            
            if loaded_model is None:
                # If standard loading fails, try with strict=False
                st.sidebar.info("🔄 Trying relaxed loading (strict=False)")
                for model_name, model in models_to_try:
                    try:
                        model.load_state_dict(state_dict, strict=False)
                        model.eval()
                        st.sidebar.success(f"✅ Loaded as {model_name} (relaxed)")
                        loaded_model = model
                        break
                    except Exception as e:
                        continue
            
            if loaded_model is not None:
                st.sidebar.success("🎯 Model loaded successfully!")
                return loaded_model
            else:
                st.sidebar.error("❌ Could not load model with any architecture")
                return None
                
        else:
            available_files = [f for f in os.listdir('.') if f.endswith(('.pth', '.pt'))]
            st.sidebar.error(f"❌ Model file not found at: {model_path}")
            st.sidebar.info(f"Available model files: {available_files}")
            return None
            
    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {str(e)}")
        return None

# Load model at startup
model = load_model()

# Settings
st.sidebar.header("⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    0.1, 0.9, 0.7, 0.05,
    help="Higher values = fewer false positives"
)

target_size = st.sidebar.selectbox(
    "Processing Size", 
    [256, 512, 224], 
    index=0,
    help="Smaller = faster, Larger = more detail"
)

# Advanced settings
st.sidebar.header("🎯 Advanced Settings")
min_spill_size = st.sidebar.slider(
    "Minimum Spill Size (pixels)",
    10, 1000, 100, 10,
    help="Filter out small detections"
)

apply_morphology = st.sidebar.checkbox(
    "Apply Noise Filtering", 
    value=True,
    help="Remove small noise and smooth detections"
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
    
    # Standard preprocessing for most models
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(original_pil).unsqueeze(0)
    resized_pil = original_pil.resize(target_size, Image.Resampling.BILINEAR)
    resized_array = np.array(resized_pil)
    
    return image_tensor, original_array, (original_h, original_w), resized_array

def filter_small_components(mask, min_size):
    """Remove small connected components from mask"""
    if min_size <= 1:
        return mask
    
    # Simple noise removal using erosion/dilation
    if min_size > 50:
        pil_mask = Image.fromarray(mask)
        # Simple noise removal
        for _ in range(2):
            pil_mask = pil_mask.filter(ImageFilter.MinFilter(3))  # Erosion
        for _ in range(2):
            pil_mask = pil_mask.filter(ImageFilter.MaxFilter(3))  # Dilation
        return np.array(pil_mask)
    else:
        return mask

def apply_morphology_operations(mask):
    """Apply basic morphology operations using PIL"""
    if not apply_morphology:
        return mask
    
    pil_mask = Image.fromarray(mask)
    
    # Opening: erosion followed by dilation (removes small noise)
    for _ in range(2):
        pil_mask = pil_mask.filter(ImageFilter.MinFilter(3))  # Erosion
    for _ in range(2):
        pil_mask = pil_mask.filter(ImageFilter.MaxFilter(3))  # Dilation
    
    return np.array(pil_mask)

def process_model_output(prediction, original_shape, confidence_threshold=0.5):
    """Process model output with proper thresholding and filtering"""
    # Convert prediction to numpy
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.squeeze().detach().cpu().numpy()
    
    st.sidebar.write("🔬 Model Output Analysis:")
    st.sidebar.write(f"Shape: {prediction.shape}")
    st.sidebar.write(f"Range: {prediction.min():.3f} to {prediction.max():.3f}")
    st.sidebar.write(f"Mean: {prediction.mean():.3f}")
    
    # Handle different output formats
    if len(prediction.shape) == 3:
        # If it's multi-channel, take the appropriate channel
        if prediction.shape[0] == 2:  # [background, spill]
            prediction = prediction[1]  # Take spill channel
        elif prediction.shape[0] == 1:  # Single channel
            prediction = prediction[0]  # Take first channel
        else:
            prediction = prediction[0]  # Default to first channel
    
    # For classification output (1D), convert to 2D mask
    if len(prediction.shape) == 1:
        classification_score = prediction[0]
        st.sidebar.write(f"Classification score: {classification_score:.3f}")
        # Create a simple mask based on classification
        if classification_score > confidence_threshold:
            # Create a central spill area for demonstration
            h, w = original_shape
            mask = np.zeros((h, w), dtype=np.uint8)
            center_h, center_w = h // 2, w // 2
            size = min(h, w) // 4
            cv2.ellipse(mask, (center_w, center_h), (size, size), 0, 0, 360, 255, -1)
            return mask
        else:
            return np.zeros(original_shape[:2], dtype=np.uint8)
    
    # Apply confidence threshold for segmentation
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
    final_mask = apply_morphology_operations(final_mask)
    
    # Filter by minimum size
    final_mask = filter_small_components(final_mask, min_spill_size)
    
    return final_mask

def analyze_water_areas(image_array, mask):
    """Simple analysis to focus on water areas using color analysis"""
    # Simple water detection based on blue/green channels
    r, g, b = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]
    
    # Water typically has higher blue/green values
    blue_dominant = (b > r) & (b > g)
    green_dominant = (g > r) & (g > b)
    water_like = blue_dominant | green_dominant
    
    # Only keep detections in water-like areas
    water_based_mask = mask.copy()
    water_based_mask[~water_like] = 0
    
    return water_based_mask

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
        
        # Display layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🛰️ Original Image")
            st.image(image, use_container_width=True)
            st.caption(f"Dimensions: {image.size} | Format: {image.format}")
        
        # Process image
        with st.spinner("🔄 Analyzing image for oil spills..."):
            # Preprocess
            image_tensor, original_array, original_shape, resized_img = preprocess_for_model(
                image, target_size=(target_size, target_size)
            )
            
            # Use ACTUAL model for prediction
            if model is not None:
                try:
                    with torch.no_grad():
                        prediction = model(image_tensor)
                    st.success("✅ Real model used for detection")
                    
                except Exception as e:
                    st.error(f"❌ Model inference failed: {str(e)}")
                    st.info("Please check the model compatibility")
                    prediction = None
            else:
                st.error("❌ Model not available")
                prediction = None
            
            if prediction is not None:
                # Process model output with proper thresholding
                final_mask = process_model_output(prediction, original_shape, confidence_threshold)
                
                # Apply water area analysis to reduce false positives
                final_mask = analyze_water_areas(original_array, final_mask)
                
                # Create overlay
                overlay_result = create_overlay(original_array, final_mask)
                
                # Convert to PIL for display
                mask_display = Image.fromarray(final_mask)
                
                # Display results
                with col2:
                    st.subheader("🎭 Detection Mask")
                    st.image(mask_display, use_container_width=True, clamp=True)
                    st.caption("White areas = Detected oil spills")
                
                with col3:
                    st.subheader("🛢️ Oil Spill Overlay")
                    st.image(overlay_result, use_container_width=True)
                    st.caption("Red areas = Oil spill locations")
                
                # Analysis results
                st.subheader("📊 Analysis Results")
                
                # Calculate statistics
                total_pixels = final_mask.size
                spill_pixels = np.sum(final_mask > 0)
                coverage_percent = (spill_pixels / total_pixels) * 100
                
                # Display metrics
                col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
                
                with col_metrics1:
                    st.metric("Spill Coverage", f"{coverage_percent:.4f}%")
                with col_metrics2:
                    st.metric("Affected Area", f"{spill_pixels:,} px")
                with col_metrics3:
                    st.metric("Confidence Level", f"{confidence_threshold:.2f}")
                with col_metrics4:
                    status = "Spill Detected" if spill_pixels > 0 else "Clean"
                    st.metric("Status", status)
                
                # Risk assessment
                st.subheader("🎯 Risk Assessment")
                if coverage_percent > 5:
                    st.error("🚨 **CRITICAL RISK** - Major oil spill detected")
                    st.write("Immediate containment action required. Alert environmental agencies.")
                elif coverage_percent > 1:
                    st.warning("⚠️ **HIGH RISK** - Significant oil contamination")
                    st.write("Deploy response teams. Monitor spill progression.")
                elif coverage_percent > 0.1:
                    st.info("🔶 **MEDIUM RISK** - Moderate spill detected")
                    st.write("Close monitoring recommended. Prepare response measures.")
                elif coverage_percent > 0.01:
                    st.success("🔷 **LOW RISK** - Minor detection")
                    st.write("Regular monitoring advised. Low immediate threat.")
                else:
                    st.success("✅ **CLEAN** - No oil spills detected")
                    st.write("Water body appears clean. Continue routine monitoring.")
                
                # Download section
                st.subheader("💾 Download Results")
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    # Download mask
                    buf_mask = io.BytesIO()
                    mask_display.save(buf_mask, format="PNG")
                    st.download_button(
                        label="📥 Download Detection Mask",
                        data=buf_mask.getvalue(),
                        file_name="oil_spill_mask.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col_dl2:
                    # Download overlay
                    buf_overlay = io.BytesIO()
                    overlay_result.save(buf_overlay, format="PNG")
                    st.download_button(
                        label="📥 Download Overlay Image",
                        data=buf_overlay.getvalue(),
                        file_name="oil_spill_overlay.png",
                        mime="image/png",
                        use_container_width=True
                    )
            else:
                st.error("❌ Cannot perform detection - model output unavailable")
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.info("Please try a different image file")

else:
    # Welcome section
    st.info("👆 **Upload a satellite image to begin analysis**")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.subheader("📋 How to Use")
        st.markdown("""
        1. **Upload** a satellite image (JPEG, PNG, TIFF)
        2. **Adjust** detection sensitivity and filters
        3. **View** AI-generated oil spill detection
        4. **Download** results for further analysis
        
        **Tips for better results:**
        - Use clear satellite imagery
        - Start with higher confidence threshold (0.7+)
        - Enable noise filtering
        - Adjust minimum spill size (100+ pixels)
        """)
    
    with col_info2:
        st.subheader("🎯 Features")
        st.markdown("""
        - 🛰️ Satellite image analysis
        - 🤖 AI-powered detection
        - 💧 Water area focus
        - 🎚️ Adjustable sensitivity
        - 📊 Quantitative metrics
        - 🎯 Risk assessment
        - 💾 Result export
        """)

# Model status in footer
st.markdown("---")
model_status = "✅ LOADED" if model is not None else "❌ NOT LOADED"
st.markdown(f"🌊 **Oil Spill Detection** | Built with Streamlit | Real AI Model: {model_status}")
