import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2
import segmentation_models_pytorch as smp
import gdown

# ------------------------
# Model configuration
# ------------------------
MODEL_PATH = "oil_spill_model_deploy.pth"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/stl47n6ixrzv59xs2jt4m/oil_spill_model_deploy.pth?rlkey=rojyk0fq73mk8tai8jc3exrev&st=w6qm08lh&dl=1"

# ------------------------
# Download model from Dropbox
# ------------------------
if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading model from Dropbox, please wait...")
    try:
        gdown.download(DROPBOX_URL, MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Download failed: {e}")
        st.info("Please check the Dropbox link or internet connection.")

# ------------------------
# Load model
# ------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Please check the download.")
        return None
    
    try:
        # Create U-Net model with ResNet34 encoder
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
        
        # Load the trained weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load the model
model = load_model()

# ------------------------
# Preprocess image
# ------------------------
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)

# ------------------------
# Postprocess output mask
# ------------------------
def postprocess_output(output_tensor, original_size, threshold=0.5):
    # Get probability mask
    probability_mask = torch.sigmoid(output_tensor).detach().cpu().numpy()[0, 0]
    
    # Create binary mask
    binary_mask = (probability_mask > threshold).astype(np.uint8)
    
    # Resize to original image size
    binary_mask_resized = cv2.resize(binary_mask, original_size)
    
    return binary_mask_resized, probability_mask

# ------------------------
# Create overlay with black oil spills
# ------------------------
def create_black_overlay(original_image, mask):
    """Create result image with oil spills in black"""
    # Convert to numpy if needed
    if isinstance(original_image, Image.Image):
        original_np = np.array(original_image)
    else:
        original_np = original_image.copy()
    
    # Ensure mask matches original size
    if original_np.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (original_np.shape[1], original_np.shape[0]))
    
    # Create result image
    result = original_np.copy()
    
    # Set oil spill areas to BLACK
    if len(result.shape) == 3:  # Color image
        result[mask == 1] = [0, 0, 0]  # RGB black
    else:  # Grayscale image
        result[mask == 1] = 0  # Black
    
    return result

# ------------------------
# Streamlit interface
# ------------------------
st.set_page_config(page_title="Oil Spill Detection", layout="wide")
st.title("🛢️ Oil Spill Detection System")
st.write("Upload a satellite image to detect oil spill regions.")

# Sidebar for settings
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

# Model info
if model is not None:
    st.sidebar.success("✅ Model ready for detection")
    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    st.sidebar.info(f"Model size: {file_size:.1f} MB")
else:
    st.sidebar.error("❌ Model not loaded")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write(f"Image size: {image.size}")
    
    with col2:
        if model is not None:
            if st.button("Detect Oil Spills", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Preprocess image
                        input_tensor = preprocess_image(image)
                        
                        # Run model prediction
                        with torch.no_grad():
                            output = model(input_tensor)
                        
                        # Postprocess output
                        binary_mask, probability_mask = postprocess_output(
                            output, image.size, confidence_threshold
                        )
                        
                        # Create result with black oil spills
                        result_image = create_black_overlay(image, binary_mask)
                        
                        # Display results
                        st.image(result_image, caption=f"Oil Spills in Black (Threshold: {confidence_threshold})", use_column_width=True)
                        
                        # Calculate statistics
                        oil_pixels = np.sum(binary_mask)
                        total_pixels = binary_mask.size
                        oil_percentage = (oil_pixels / total_pixels) * 100
                        
                        st.success("✅ Detection completed!")
                        
                        # Show statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Oil Coverage", f"{oil_percentage:.2f}%")
                        with col2:
                            st.metric("Oil Pixels", f"{oil_pixels:,}")
                        with col3:
                            st.metric("Confidence", f"{confidence_threshold}")
                        
                        # Save option
                        if st.checkbox("Save result image"):
                            result_pil = Image.fromarray(result_image)
                            result_pil.save("oil_spill_result.png")
                            st.success("💾 Result saved as 'oil_spill_result.png'")
                            
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
        else:
            st.error("Model not loaded. Please check the download status.")

# ------------------------
# Instructions
# ------------------------
with st.expander("ℹ️ Instructions"):
    st.markdown("""
    **How to use:**
    1. The app will automatically download the model from Dropbox
    2. Upload a satellite image (JPG, PNG, JPEG)
    3. Click 'Detect Oil Spills' button
    4. View results with oil spills shown in black
    5. Adjust confidence threshold if needed
    
    **Model Information:**
    - Architecture: U-Net with ResNet34 encoder
    - Training: 25 epochs on oil spill dataset
    - Performance: 79.5% IoU, 82.6% Precision
    
    **Visualization:**
    - Oil spills appear as **black regions**
    - Original image background remains unchanged
    - Adjust threshold for more/less sensitive detection
    """)

# Footer
st.markdown("---")
st.markdown("**AI Oil Spill Detection System** | Model: U-Net ResNet34 | Oil spills shown in black")