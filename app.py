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
# ENHANCED DETECTION FUNCTIONS - KEEPING WHAT WORKS
# -----------------------------
def enhanced_oil_detection(model, image, device, confidence_threshold=0.5):
    """Enhanced detection with multiple strategies for difficult images"""
    
    strategies = []
    
    # Strategy 1: Original image
    input_tensor, processed_img = preprocess_image(image)
    input_tensor = input_tensor.to(device, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        pred1 = torch.sigmoid(output).squeeze().cpu().numpy()
    strategies.append(("Standard", pred1))
    
    # Strategy 2: High contrast
    high_contrast = ImageEnhance.Contrast(image).enhance(2.0)
    input_tensor, _ = preprocess_image(high_contrast)
    input_tensor = input_tensor.to(device, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        pred2 = torch.sigmoid(output).squeeze().cpu().numpy()
    strategies.append(("High Contrast", pred2))
    
    # Strategy 3: Sharpened
    sharpened = image.filter(ImageFilter.SHARPEN)
    input_tensor, _ = preprocess_image(sharpened)
    input_tensor = input_tensor.to(device, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        pred3 = torch.sigmoid(output).squeeze().cpu().numpy()
    strategies.append(("Sharpened", pred3))
    
    # Strategy 4: Brightness adjusted
    bright = ImageEnhance.Brightness(image).enhance(1.4)
    input_tensor, _ = preprocess_image(bright)
    input_tensor = input_tensor.to(device, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        pred4 = torch.sigmoid(output).squeeze().cpu().numpy()
    strategies.append(("Brightened", pred4))
    
    # Strategy 5: Color enhanced
    colorful = ImageEnhance.Color(image).enhance(1.5)
    input_tensor, _ = preprocess_image(colorful)
    input_tensor = input_tensor.to(device, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        pred5 = torch.sigmoid(output).squeeze().cpu().numpy()
    strategies.append(("Color Enhanced", pred5))
    
    # Let user choose or use ensemble
    st.sidebar.subheader("🎛️ Enhanced Detection")
    strategy = st.sidebar.selectbox(
        "Detection Method:",
        ["Smart Ensemble", "Standard", "High Contrast", "Sharpened", "Brightened", "Color Enhanced"]
    )
    
    if strategy == "Smart Ensemble":
        # Weighted average based on confidence
        all_preds = [pred for _, pred in strategies]
        ensemble_pred = np.mean(all_preds, axis=0)
        st.success("🤖 Using Smart Ensemble (recommended for difficult images)")
        return ensemble_pred, processed_img
    else:
        # Use selected strategy
        for name, pred in strategies:
            if name == strategy:
                st.success(f"🔧 Using {strategy} method")
                return pred, processed_img
    
    return pred1, processed_img  # fallback

# -----------------------------
# FIXED OVERLAY FUNCTION - KEEPING YOUR WORKING APPROACH
# -----------------------------
def create_proper_overlay(processed_image, binary_mask):
    """Create overlay using your working approach"""
    overlay = processed_image.copy()
    overlay_np = np.array(overlay)
    
    # Apply red color to detected areas - EXACTLY like your working code
    overlay_np[binary_mask > 0] = [255, 0, 0]  # red for oil spill
    
    overlay_img = Image.fromarray(overlay_np)
    return overlay_img

def debug_prediction_info(prediction, binary_mask, confidence_threshold):
    """Debug information to see what's happening"""
    st.subheader("🔍 Debug Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Prediction Stats:**")
        st.write(f"Min: {prediction.min():.6f}")
        st.write(f"Max: {prediction.max():.6f}")
        st.write(f"Mean: {prediction.mean():.6f}")
        st.write(f"Std: {prediction.std():.6f}")
    
    with col2:
        st.write("**Threshold Analysis:**")
        st.write(f"Threshold: {confidence_threshold:.6f}")
        st.write(f"Pixels above: {np.sum(prediction > confidence_threshold)}")
        st.write(f"Total pixels: {prediction.size}")
        st.write(f"Percentage: {np.sum(prediction > confidence_threshold) / prediction.size * 100:.6f}%")
    
    with col3:
        st.write("**Mask Info:**")
        st.write(f"Mask sum: {np.sum(binary_mask > 0)}")
        st.write(f"Mask unique values: {np.unique(binary_mask)}")
        st.write(f"Non-zero pixels: {np.count_nonzero(binary_mask)}")

# -----------------------------
# Streamlit App - USING YOUR WORKING STRUCTURE
# -----------------------------
st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊", layout="wide")
st.title("🌊 Oil Spill Detection - Fixed Version")
st.write("Upload a satellite image to detect oil spills.")

with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.001, 0.9, 0.1, 0.001)
    
    st.header("🎛️ Detection Mode")
    detection_mode = st.radio(
        "Choose detection mode:",
        ["Standard", "Enhanced"]
    )
    
    show_debug = st.checkbox("Show Debug Info", value=True)
    
    st.header("ℹ️ Tips")
    st.write("• Use very low thresholds (0.001-0.01) for faint spills")
    st.write("• Enhanced mode works better for difficult images")

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
        with st.spinner("🔄 Running detection..."):
            if detection_mode == "Enhanced":
                # Use enhanced detection
                prediction, processed_image = enhanced_oil_detection(
                    st.session_state.model, 
                    image, 
                    st.session_state.device,
                    confidence_threshold
                )
            else:  # Standard mode
                input_tensor, processed_image = preprocess_image(image)
                input_tensor = input_tensor.to(st.session_state.device, dtype=torch.float32)
                with torch.no_grad():
                    output = st.session_state.model(input_tensor)
                    prediction = torch.sigmoid(output).squeeze().cpu().numpy()

            # Binary mask - USING YOUR WORKING APPROACH
            binary_mask = (prediction > confidence_threshold).astype(np.uint8) * 255

            # Overlay mask on original - USING YOUR WORKING APPROACH
            overlay_img = create_proper_overlay(processed_image, binary_mask)

            # Display results - KEEPING YOUR WORKING LAYOUT
            col1, col2 = st.columns(2)
            col1.image(processed_image, caption="Processed Image", use_column_width=True)
            col2.image(overlay_img, caption="Oil Spill Overlay", use_column_width=True)

            # Metrics - USING YOUR WORKING CALCULATIONS
            spill_pixels = np.sum(binary_mask > 0)
            total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
            spill_area = (spill_pixels / total_pixels) * 100
            max_conf = np.max(prediction) * 100
            mean_conf = np.mean(prediction) * 100

            st.subheader("📊 Detection Results")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Spill Area", f"{spill_area:.4f}%")
            col2.metric("Spill Pixels", f"{spill_pixels}")
            col3.metric("Max Confidence", f"{max_conf:.2f}%")
            col4.metric("Mean Confidence", f"{mean_conf:.2f}%")
            
            # Status determination - MORE SENSITIVE
            status = "🔴 Spill Detected" if spill_pixels > 0 else "🟢 No Spill"
            st.metric("Status", status)

            # Show the actual mask
            st.subheader("🎭 Detection Mask")
            mask_image = Image.fromarray(binary_mask)
            st.image(mask_image, caption="Binary Detection Mask (White = Oil Spill)", use_column_width=True, clamp=True)

            # Download mask
            buf = io.BytesIO()
            mask_image.save(buf, format="PNG")
            st.download_button(
                label="💾 Download Prediction Mask",
                data=buf.getvalue(),
                file_name="oil_spill_mask.png",
                mime="image/png"
            )

            # Debug information
            if show_debug:
                debug_prediction_info(prediction, binary_mask, confidence_threshold)

                # Show what happens with different thresholds
                st.subheader("🎯 Threshold Testing")
                test_thresholds = [0.001, 0.01, 0.05, 0.1, 0.2]
                cols = st.columns(len(test_thresholds))
                
                for idx, test_thresh in enumerate(test_thresholds):
                    with cols[idx]:
                        test_mask = (prediction > test_thresh).astype(np.uint8) * 255
                        test_overlay = create_proper_overlay(processed_image, test_mask)
                        st.image(test_overlay, caption=f"Thresh: {test_thresh:.3f}", use_column_width=True)
                        test_area = np.sum(test_mask > 0) / test_mask.size * 100
                        st.write(f"Area: {test_area:.4f}%")

else:
    st.info("👆 Please upload a satellite image to begin detection.")
