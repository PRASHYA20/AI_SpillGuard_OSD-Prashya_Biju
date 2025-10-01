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
# FIXED OVERLAY FUNCTIONS
# -----------------------------
def create_better_overlay(original_image, mask, alpha=0.7):
    """Create a more visible overlay"""
    # Ensure mask is the right size
    if original_image.size != mask.size:
        mask = mask.resize(original_image.size, Image.NEAREST)
    
    # Convert to RGBA
    original_rgba = original_image.convert('RGBA')
    
    # Create red overlay with transparency
    red_overlay = Image.new('RGBA', original_rgba.size, (255, 0, 0, int(255 * alpha)))
    
    # Convert mask to binary and ensure it's the right mode
    mask_binary = np.array(mask) > 0
    mask_pil = Image.fromarray((mask_binary * 255).astype(np.uint8)).convert('L')
    
    # Composite the images
    result = Image.composite(red_overlay, original_rgba, mask_pil)
    return result.convert('RGB')

def create_enhanced_visualization(original_image, prediction, confidence_threshold):
    """Create multiple visualization options"""
    # Create binary mask
    binary_mask = (prediction > confidence_threshold).astype(np.uint8) * 255
    mask_image = Image.fromarray(binary_mask)
    
    # Resize mask to match original image size
    if original_image.size != (256, 256):
        mask_resized = mask_image.resize(original_image.size, Image.NEAREST)
    else:
        mask_resized = mask_image
    
    # Create different overlay types
    overlay_simple = create_better_overlay(original_image, mask_resized, alpha=0.7)
    overlay_strong = create_better_overlay(original_image, mask_resized, alpha=0.9)
    
    # Create outline overlay (just the edges)
    edges = mask_image.filter(ImageFilter.FIND_EDGES)
    edges_resized = edges.resize(original_image.size, Image.NEAREST)
    overlay_edges = create_better_overlay(original_image, edges_resized, alpha=0.8)
    
    return overlay_simple, overlay_strong, overlay_edges, mask_resized

def debug_detection_details(prediction, binary_mask, confidence_threshold):
    """Show detailed debug information"""
    st.subheader("🔍 Detection Debug Info")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Prediction Statistics:**")
        st.write(f"Min: {prediction.min():.4f}")
        st.write(f"Max: {prediction.max():.4f}")
        st.write(f"Mean: {prediction.mean():.4f}")
        st.write(f"Std: {prediction.std():.4f}")
    
    with col2:
        st.write("**Threshold Analysis:**")
        st.write(f"Threshold: {confidence_threshold:.3f}")
        st.write(f"Pixels above: {np.sum(prediction > confidence_threshold)}")
        st.write(f"Total pixels: {prediction.size}")
        st.write(f"Percentage: {np.sum(prediction > confidence_threshold) / prediction.size * 100:.2f}%")
    
    with col3:
        st.write("**Mask Analysis:**")
        st.write(f"Mask sum: {np.sum(binary_mask > 0)}")
        st.write(f"Mask unique: {np.unique(binary_mask)}")
        st.write(f"Mask shape: {binary_mask.shape}")
        st.write(f"Mask dtype: {binary_mask.dtype}")

# -----------------------------
# ENHANCED DETECTION FUNCTIONS
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
    
    # Let user choose or use ensemble
    st.sidebar.subheader("🎛️ Enhanced Detection")
    strategy = st.sidebar.selectbox(
        "Detection Method:",
        ["Smart Ensemble", "Standard", "High Contrast", "Sharpened"]
    )
    
    if strategy == "Smart Ensemble":
        # Weighted average based on confidence
        all_preds = [pred for _, pred in strategies]
        ensemble_pred = np.mean(all_preds, axis=0)
        st.success("🤖 Using Smart Ensemble")
        return ensemble_pred, processed_img
    else:
        # Use selected strategy
        for name, pred in strategies:
            if name == strategy:
                st.success(f"🔧 Using {strategy} method")
                return pred, processed_img
    
    return pred1, processed_img  # fallback

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊", layout="wide")
st.title("🌊 Oil Spill Detection - Fixed Overlay")
st.write("Upload a satellite image to detect oil spills.")

with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.01, 0.9, 0.1, 0.01)
    
    st.header("🎛️ Display Options")
    show_debug = st.checkbox("Show Debug Information", value=True)
    overlay_style = st.selectbox(
        "Overlay Style:",
        ["Standard Red", "Strong Red", "Edge Outline"]
    )
    
    st.header("ℹ️ About")
    st.write("This app uses a UNet model for oil spill segmentation.")

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
        with st.spinner("🔄 Running enhanced detection..."):
            # Use enhanced detection
            prediction, processed_image = enhanced_oil_detection(
                st.session_state.model, 
                image, 
                st.session_state.device,
                confidence_threshold
            )

            # Create binary mask
            binary_mask = (prediction > confidence_threshold).astype(np.uint8) * 255
            
            # Create multiple overlay options
            overlay_standard, overlay_strong, overlay_edges, mask_display = create_enhanced_visualization(
                processed_image, prediction, confidence_threshold
            )
            
            # Choose overlay based on user selection
            if overlay_style == "Strong Red":
                overlay_img = overlay_strong
            elif overlay_style == "Edge Outline":
                overlay_img = overlay_edges
            else:
                overlay_img = overlay_standard

            # Display results in a better layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🖼️ Processed Image")
                st.image(processed_image, use_column_width=True)
            
            with col2:
                st.subheader("🎭 Detection Mask")
                st.image(mask_display, use_column_width=True, clamp=True)
                st.caption("White = Detected Oil Spill")
            
            with col3:
                st.subheader("🛢️ Oil Spill Overlay")
                st.image(overlay_img, use_column_width=True)
                st.caption(f"{overlay_style}")

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
                st.metric("Spill Pixels", f"{spill_pixels:,}")
            with col3:
                st.metric("Max Confidence", f"{max_conf:.1f}%")
            with col4:
                status = "🔴 Spill Detected" if spill_pixels > 0 else "🟢 No Spill"
                st.metric("Status", status)

            # Show warning if spill detected but area is very small
            if spill_pixels > 0 and spill_area < 0.01:
                st.warning("⚠️ Very small spill detected. Try lowering the confidence threshold to see more details.")

            # Debug information
            if show_debug:
                debug_detection_details(prediction, binary_mask, confidence_threshold)

            # Download options
            st.subheader("💾 Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                # Download mask
                buf_mask = io.BytesIO()
                mask_display.save(buf_mask, format="PNG")
                st.download_button(
                    label="📥 Download Detection Mask",
                    data=buf_mask.getvalue(),
                    file_name="oil_spill_mask.png",
                    mime="image/png",
                    use_column_width=True
                )
            
            with col2:
                # Download overlay
                buf_overlay = io.BytesIO()
                overlay_img.save(buf_overlay, format="PNG")
                st.download_button(
                    label="📥 Download Overlay Image",
                    data=buf_overlay.getvalue(),
                    file_name="oil_spill_overlay.png",
                    mime="image/png",
                    use_column_width=True
                )

else:
    st.info("👆 Please upload a satellite image to begin detection.")
