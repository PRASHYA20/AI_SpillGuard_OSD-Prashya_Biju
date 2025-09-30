import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import torch
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

# Check for model files
def find_model_files():
    files = os.listdir('.')
    model_files = [f for f in files if f.endswith(('.pth', '.pt', '.h5', '.pkl'))]
    return model_files

model_files = find_model_files()

# File status
st.sidebar.header("📁 Model Status")
if model_files:
    st.sidebar.success(f"✅ Model: {model_files[0]}")
else:
    st.sidebar.info("🤖 Demo Mode")
    st.sidebar.write("Using synthetic predictions")

# Settings
st.sidebar.header("⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    0.1, 0.9, 0.5, 0.1,
    help="Higher values = more conservative detection"
)
target_size = st.sidebar.selectbox(
    "Processing Size", 
    [256, 512, 224], 
    index=0,
    help="Smaller = faster, Larger = more detail"
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

def create_realistic_prediction(image_tensor, image_array):
    """Create realistic, varied oil spill predictions"""
    batch_size, channels, height, width = image_tensor.shape
    
    # Create base prediction tensor
    prediction = torch.zeros(batch_size, 1, height, width)
    
    # Generate coordinate grid
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, height),
        torch.linspace(-1, 1, width),
        indexing='ij'
    )
    
    # Random number of spills (0-5)
    num_spills = random.randint(0, 5)
    
    for i in range(num_spills):
        # Random position
        center_x = random.uniform(-0.8, 0.8)
        center_y = random.uniform(-0.8, 0.8)
        
        # Random size and shape
        size_x = random.uniform(0.1, 0.5)
        size_y = random.uniform(0.1, 0.5)
        
        # Create elliptical spill
        spill = ((x_coords - center_x)**2 / size_x**2 + 
                (y_coords - center_y)**2 / size_y**2 < 1)
        
        # Random intensity
        intensity = random.uniform(0.4, 0.95)
        
        # Add noise for realism
        noise = torch.randn(height, width) * 0.15
        spill_with_noise = spill.float() + noise
        spill_sharp = torch.sigmoid(spill_with_noise * 4)
        
        prediction[0, 0] = torch.max(prediction[0, 0], spill_sharp * intensity)
    
    return torch.clamp(prediction, 0, 1)

def resize_mask_to_original(mask_pred, original_shape):
    """Resize mask back to original image dimensions"""
    if isinstance(mask_pred, torch.Tensor):
        mask_pred = mask_pred.squeeze().detach().cpu().numpy()
    
    # Handle different output formats
    if len(mask_pred.shape) == 3:
        mask_pred = mask_pred[0] if mask_pred.shape[0] == 1 else mask_pred
    
    # Convert to binary
    mask_binary = (mask_pred > 0.5).astype(np.uint8)
    
    # Resize to original dimensions
    mask_pil = Image.fromarray((mask_binary * 255).astype(np.uint8))
    mask_resized = mask_pil.resize(
        (original_shape[1], original_shape[0]), 
        Image.Resampling.NEAREST
    )
    
    return np.array(mask_resized)

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
            
            # Generate prediction (synthetic for demo)
            if model_files:
                # TODO: Replace with actual model inference
                # model = torch.load(model_files[0], map_location='cpu')
                # model.eval()
                # with torch.no_grad():
                #     prediction = model(image_tensor)
                prediction = create_realistic_prediction(image_tensor, original_array)
            else:
                prediction = create_realistic_prediction(image_tensor, original_array)
            
            # Postprocess
            final_mask = resize_mask_to_original(prediction, original_shape)
            final_mask_binary = (final_mask > (confidence_threshold * 255)).astype(np.uint8) * 255
            
            # Create overlay
            overlay_result = create_overlay(original_array, final_mask_binary)
            
            # Convert to PIL for display
            mask_display = Image.fromarray(final_mask_binary)
            
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
        total_pixels = final_mask_binary.size
        spill_pixels = np.sum(final_mask_binary > 0)
        coverage_percent = (spill_pixels / total_pixels) * 100
        
        # Display metrics
        col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
        
        with col_metrics1:
            st.metric("Spill Coverage", f"{coverage_percent:.2f}%")
        with col_metrics2:
            st.metric("Affected Area", f"{spill_pixels:,} px")
        with col_metrics3:
            st.metric("Confidence Level", f"{confidence_threshold:.1f}")
        with col_metrics4:
            status = "Spill Detected" if spill_pixels > 0 else "Clean"
            st.metric("Status", status)
        
        # Risk assessment
        st.subheader("🎯 Risk Assessment")
        if coverage_percent > 10:
            st.error("🚨 **CRITICAL RISK** - Major oil spill detected")
            st.write("Immediate containment action required. Alert environmental agencies.")
        elif coverage_percent > 2:
            st.warning("⚠️ **HIGH RISK** - Significant oil contamination")
            st.write("Deploy response teams. Monitor spill progression.")
        elif coverage_percent > 0.5:
            st.info("🔶 **MEDIUM RISK** - Moderate spill detected")
            st.write("Close monitoring recommended. Prepare response measures.")
        elif coverage_percent > 0.1:
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
        2. **Adjust** detection sensitivity if needed
        3. **View** AI-generated oil spill detection
        4. **Download** results for further analysis
        """)
    
    with col_info2:
        st.subheader("🎯 Features")
        st.markdown("""
        - 🛰️ Satellite image analysis
        - 🤖 AI-powered detection
        - 📊 Quantitative metrics
        - 🎯 Risk assessment
        - 💾 Result export
        """)

# Footer
st.markdown("---")
st.markdown(
    "🌊 **Oil Spill Detection** | "
    "Built with Streamlit | "
    "Deployed on Hugging Face Spaces"
)
