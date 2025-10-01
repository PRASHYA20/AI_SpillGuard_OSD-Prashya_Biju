import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io
import requests
import os
import matplotlib.pyplot as plt

# -----------------------------
# Dropbox Model URL
# -----------------------------
MODEL_PATH = "oil_spill_model_deploy.pth"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/stl47n6ixrzv59xs2jt4m/oil_spill_model_deploy.pth?rlkey=rojyk0fq73mk8tai8jc3exrev&dl=1"

# -----------------------------
# Create UNet Model
# -----------------------------
def create_unet_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model

# -----------------------------
# Download model if missing
# -----------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("🔽 Downloading model from Dropbox...")
        try:
            response = requests.get(DROPBOX_URL, stream=True, timeout=60)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = st.progress(0)
            downloaded = 0
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(progress)
            progress_bar.empty()
            st.success("✅ Model downloaded!")
        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            return False
    return True

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"🖥️ Using device: {device}")

    if not download_model():
        return None, device

    try:
        model = create_unet_model()
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Remove DataParallel prefix if exists
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        st.success("✅ Model loaded successfully!")
        return model, device

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, device

# -----------------------------
# Preprocess Image
# -----------------------------
def preprocess_image(image):
    image_resized = image.resize((256, 256))
    img_array = np.array(image_resized).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    # Float32 tensor
    img_tensor = torch.from_numpy(img_array).permute(2,0,1).unsqueeze(0).float()
    return img_tensor, image_resized

# -----------------------------
# SIMPLIFIED ENHANCED DETECTION - NO EXTERNAL DEPENDENCIES
# -----------------------------
def enhanced_oil_detection(model, image, device, confidence_threshold=0.5):
    """Enhanced detection with multiple strategies"""
    
    strategies = []
    
    # Strategy 1: Original image
    input_tensor, processed_img = preprocess_image(image)
    input_tensor = input_tensor.to(device, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        pred1 = torch.sigmoid(output).squeeze().cpu().numpy()
    strategies.append(("Standard", pred1, processed_img))
    
    # Strategy 2: High contrast
    high_contrast = ImageEnhance.Contrast(image).enhance(2.0)
    input_tensor, processed_img_hc = preprocess_image(high_contrast)
    input_tensor = input_tensor.to(device, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        pred2 = torch.sigmoid(output).squeeze().cpu().numpy()
    strategies.append(("High Contrast", pred2, processed_img_hc))
    
    # Strategy 3: Sharpened
    sharpened = image.filter(ImageFilter.SHARPEN)
    input_tensor, processed_img_sharp = preprocess_image(sharpened)
    input_tensor = input_tensor.to(device, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        pred3 = torch.sigmoid(output).squeeze().cpu().numpy()
    strategies.append(("Sharpened", pred3, processed_img_sharp))
    
    return strategies

def simple_adaptive_threshold(prediction):
    """Simple adaptive threshold without scikit-image"""
    # Use mean + standard deviation as threshold
    threshold = np.mean(prediction) + np.std(prediction)
    return threshold

# -----------------------------
# FIXED OVERLAY FUNCTION
# -----------------------------
def create_overlay(processed_image, binary_mask):
    """Create overlay that actually works"""
    overlay = processed_image.copy()
    overlay_np = np.array(overlay)
    
    # Apply red color to detected areas
    overlay_np[binary_mask > 0] = [255, 0, 0]  # red for oil spill
    
    return Image.fromarray(overlay_np)

# -----------------------------
# Streamlit App - SIMPLIFIED AND WORKING
# -----------------------------
st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊", layout="wide")
st.title("🌊 Oil Spill Detection")
st.write("Upload a satellite image to detect oil spills.")

with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.001, 0.9, 0.05, 0.001)
    
    st.header("🎛️ Detection Mode")
    detection_mode = st.radio(
        "Choose detection mode:",
        ["Standard", "Enhanced", "Compare All"]
    )
    
    st.header("ℹ️ Tips")
    st.write("• Start with low threshold (0.01-0.1)")
    st.write("• Enhanced mode for difficult images")
    st.write("• Compare All to see all methods")

# Initialize model
if 'model' not in st.session_state:
    with st.spinner("🔄 Loading model..."):
        model, device = load_model()
        st.session_state.model = model
        st.session_state.device = device

# Upload image
uploaded_file = st.file_uploader("📤 Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    if st.session_state.model is None:
        st.error("❌ Model failed to load. Please check weights.")
    else:
        if detection_mode == "Compare All":
            # Compare all strategies
            st.subheader("🔍 Comparing All Detection Methods")
            
            strategies = enhanced_oil_detection(
                st.session_state.model, 
                image, 
                st.session_state.device,
                confidence_threshold
            )
            
            # Display all strategies
            cols = st.columns(3)
            for idx, (name, prediction, processed_img) in enumerate(strategies):
                with cols[idx]:
                    # Create binary mask
                    binary_mask = (prediction > confidence_threshold).astype(np.uint8) * 255
                    
                    # Create overlay
                    overlay_img = create_overlay(processed_img, binary_mask)
                    
                    # Display
                    st.image(overlay_img, caption=name, use_column_width=True)
                    
                    # Metrics
                    spill_pixels = np.sum(binary_mask > 0)
                    total_pixels = binary_mask.size
                    spill_area = (spill_pixels / total_pixels) * 100
                    
                    st.write(f"**Spill Area:** {spill_area:.4f}%")
                    st.write(f"**Pixels:** {spill_pixels}")
                    
        else:
            with st.spinner("🔄 Running detection..."):
                if detection_mode == "Enhanced":
                    # Use the best enhanced strategy
                    strategies = enhanced_oil_detection(
                        st.session_state.model, 
                        image, 
                        st.session_state.device,
                        confidence_threshold
                    )
                    
                    # Use High Contrast by default for enhanced mode
                    prediction, processed_image = strategies[1][1], strategies[1][2]  # High Contrast
                    st.success("🔧 Using High Contrast enhanced detection")
                    
                else:  # Standard mode
                    input_tensor, processed_image = preprocess_image(image)
                    input_tensor = input_tensor.to(st.session_state.device, dtype=torch.float32)
                    with torch.no_grad():
                        output = st.session_state.model(input_tensor)
                        prediction = torch.sigmoid(output).squeeze().cpu().numpy()

                # Binary mask
                binary_mask = (prediction > confidence_threshold).astype(np.uint8) * 255

                # Create overlay
                overlay_img = create_overlay(processed_image, binary_mask)

                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("🖼️ Processed")
                    st.image(processed_image, use_column_width=True)
                
                with col2:
                    st.subheader("🎭 Detection Mask")
                    mask_display = Image.fromarray(binary_mask)
                    st.image(mask_display, use_column_width=True, clamp=True)
                    st.caption("White = Oil Spill")
                
                with col3:
                    st.subheader("🛢️ Overlay")
                    st.image(overlay_img, use_column_width=True)
                    st.caption("Red = Detected Oil")

                # Metrics
                spill_pixels = np.sum(binary_mask > 0)
                total_pixels = binary_mask.size
                spill_area = (spill_pixels / total_pixels) * 100
                max_conf = np.max(prediction) * 100
                mean_conf = np.mean(prediction) * 100

                st.subheader("📊 Detection Results")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Spill Area", f"{spill_area:.4f}%")
                with col2:
                    st.metric("Spill Pixels", f"{spill_pixels}")
                with col3:
                    st.metric("Max Confidence", f"{max_conf:.1f}%")
                with col4:
                    status = "🔴 Spill Detected" if spill_pixels > 0 else "🟢 No Spill"
                    st.metric("Status", status)

                # Debug info
                st.subheader("🔍 Detection Details")
                debug_col1, debug_col2 = st.columns(2)
                
                with debug_col1:
                    st.write("**Prediction Statistics:**")
                    st.write(f"Min: {prediction.min():.6f}")
                    st.write(f"Max: {prediction.max():.6f}")
                    st.write(f"Mean: {prediction.mean():.6f}")
                    st.write(f"Threshold: {confidence_threshold:.6f}")
                
                with debug_col2:
                    st.write("**Detection Info:**")
                    st.write(f"Pixels above threshold: {spill_pixels}")
                    st.write(f"Total pixels: {total_pixels}")
                    st.write(f"Detection ratio: {spill_pixels/total_pixels*100:.6f}%")

                # Download options
                st.subheader("💾 Download Results")
                download_col1, download_col2 = st.columns(2)
                
                with download_col1:
                    buf_mask = io.BytesIO()
                    mask_display.save(buf_mask, format="PNG")
                    st.download_button(
                        label="📥 Download Mask",
                        data=buf_mask.getvalue(),
                        file_name="oil_spill_mask.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with download_col2:
                    buf_overlay = io.BytesIO()
                    overlay_img.save(buf_overlay, format="PNG")
                    st.download_button(
                        label="📥 Download Overlay",
                        data=buf_overlay.getvalue(),
                        file_name="oil_spill_overlay.png",
                        mime="image/png",
                        use_container_width=True
                    )

else:
    st.info("👆 Please upload a satellite image to begin detection.")
