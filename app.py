import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import io
import requests
import os
import cv2

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
            return True
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
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            # Remove DataParallel prefix if exists
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        else:
            # Assume it's a full model
            model = checkpoint
            
        model.to(device)
        model.eval()
        st.success("✅ Model loaded successfully!")
        return model, device

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, device

# -----------------------------
# Enhanced Preprocessing
# -----------------------------
def preprocess_image(image, target_size=256):
    original_size = image.size
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize maintaining aspect ratio or fixed size
    image_resized = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img_array = np.array(image_resized).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    # Convert to tensor
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor, original_size, image_resized

# -----------------------------
# Enhanced Postprocessing
# -----------------------------
def postprocess_mask(prediction, original_size, confidence_threshold=0.5):
    """Enhanced postprocessing for better mask quality"""
    # Apply threshold
    binary_mask = (prediction > confidence_threshold).astype(np.uint8) * 255
    
    # Resize to original size
    binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    binary_mask_cleaned = cv2.morphologyEx(binary_mask_resized, cv2.MORPH_OPEN, kernel)
    
    return binary_mask_cleaned, binary_mask_resized

# -----------------------------
# Create Better Overlay
# -----------------------------
def create_overlay(original_image, mask, alpha=0.6):
    """Create a professional-looking overlay"""
    original_cv = np.array(original_image)
    
    # Create colored mask (red for oil spills)
    colored_mask = np.zeros_like(original_cv)
    colored_mask[mask > 0] = [255, 0, 0]  # Red
    
    # Blend with original
    blended = cv2.addWeighted(original_cv, 1 - alpha, colored_mask, alpha, 0)
    
    return Image.fromarray(blended)

# -----------------------------
# Calculate Detailed Statistics
# -----------------------------
def calculate_statistics(mask, prediction, original_size):
    """Calculate comprehensive detection statistics"""
    total_pixels = mask.size
    oil_pixels = np.sum(mask > 0)
    coverage = (oil_pixels / total_pixels) * 100
    
    # Confidence statistics
    max_confidence = np.max(prediction) * 100
    mean_confidence = np.mean(prediction) * 100
    high_confidence_pixels = np.sum(prediction > 0.8) / prediction.size * 100
    
    # Shape analysis
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_spills = len(contours)
    
    return {
        'coverage': coverage,
        'oil_pixels': oil_pixels,
        'total_pixels': total_pixels,
        'max_confidence': max_confidence,
        'mean_confidence': mean_confidence,
        'high_confidence_pixels': high_confidence_pixels,
        'num_spills': num_spills
    }

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊", layout="wide")
st.title("🌊 Oil Spill Segmentation with UNet (ResNet34)")
st.write("Upload a satellite image to detect oil spills with high accuracy.")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Detection Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05,
                                   help="Higher values = fewer false positives, but might miss small spills")
    
    input_size = st.selectbox("Input Size", [224, 256, 384, 512], index=1,
                            help="Larger sizes may detect smaller spills better")
    
    enable_cleaning = st.checkbox("Clean Mask (Remove Noise)", value=True,
                                help="Remove small noisy detections")
    
    st.header("ℹ️ About")
    st.write("""
    This app uses a **UNet model with ResNet34 backbone** trained for oil spill segmentation.
    
    **Features:**
    - High-accuracy segmentation
    - Multiple input sizes
    - Confidence-based filtering
    - Professional visualization
    """)

# Initialize model
if 'model' not in st.session_state:
    with st.spinner("🔄 Loading segmentation model..."):
        model, device = load_model()
        st.session_state.model = model
        st.session_state.device = device

# Upload image
uploaded_file = st.file_uploader("📤 Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display original image
    image = Image.open(uploaded_file).convert("RGB")
    original_size = image.size
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🛰️ Original Image")
        st.image(image, width='stretch')
        st.write(f"Size: {original_size}")

    if st.session_state.model is None:
        st.error("❌ Model failed to load. Please check the console for details.")
    else:
        with st.spinner("🔄 Analyzing image for oil spills..."):
            try:
                # Preprocess
                input_tensor, original_size, processed_image = preprocess_image(image, input_size)
                input_tensor = input_tensor.to(st.session_state.device, dtype=torch.float32)

                # Predict
                with torch.no_grad():
                    output = st.session_state.model(input_tensor)
                    prediction = torch.sigmoid(output).squeeze().cpu().numpy()

                # Postprocess
                binary_mask, binary_mask_resized = postprocess_mask(
                    prediction, original_size, confidence_threshold
                )
                
                # Create overlay
                overlay_img = create_overlay(image, binary_mask_resized)

                # Display results
                with col2:
                    st.subheader("🎭 Prediction Mask")
                    st.image(binary_mask_resized, width='stretch', clamp=True)
                    st.write("White = Oil spill detected")

                with col3:
                    st.subheader("🛢️ Oil Spill Overlay")
                    st.image(overlay_img, width='stretch')
                    st.write("Red = Oil spill areas")

                # Calculate statistics
                stats = calculate_statistics(binary_mask_resized, prediction, original_size)

                # Display detailed analysis
                st.subheader("📊 Detailed Analysis")
                
                col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
                
                with col_metrics1:
                    st.metric("Spill Coverage", f"{stats['coverage']:.4f}%")
                with col_metrics2:
                    st.metric("Oil Pixels", f"{stats['oil_pixels']:,}")
                with col_metrics3:
                    st.metric("Max Confidence", f"{stats['max_confidence']:.1f}%")
                with col_metrics4:
                    st.metric("Detected Spills", stats['num_spills'])

                # Confidence analysis
                st.subheader("🔍 Confidence Analysis")
                conf_col1, conf_col2 = st.columns(2)
                
                with conf_col1:
                    st.write("**Confidence Distribution:**")
                    st.write(f"- Mean Confidence: {stats['mean_confidence']:.1f}%")
                    st.write(f"- High Confidence Areas: {stats['high_confidence_pixels']:.1f}%")
                    st.write(f"- Current Threshold: {confidence_threshold}")
                
                with conf_col2:
                    st.write("**Interpretation:**")
                    if stats['coverage'] > 10:
                        st.error("🚨 **LARGE OIL SPILL** - Significant area affected")
                    elif stats['coverage'] > 1:
                        st.warning("⚠️ **Medium Oil Spill** - Notable contamination")
                    elif stats['coverage'] > 0.1:
                        st.info("ℹ️ **Small Oil Spill** - Minor detection")
                    else:
                        st.success("✅ **No Significant Spill** - Clean area")
                    
                    if stats['max_confidence'] < 50:
                        st.warning("⚠️ Model uncertain - consider lowering threshold")

                # Download section
                st.subheader("💾 Download Results")
                
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                
                with col_dl1:
                    # Download mask
                    mask_image = Image.fromarray(binary_mask_resized)
                    buf_mask = io.BytesIO()
                    mask_image.save(buf_mask, format="PNG")
                    st.download_button(
                        label="📥 Download Mask",
                        data=buf_mask.getvalue(),
                        file_name="oil_spill_mask.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col_dl2:
                    # Download overlay
                    buf_overlay = io.BytesIO()
                    overlay_img.save(buf_overlay, format="PNG")
                    st.download_button(
                        label="📥 Download Overlay",
                        data=buf_overlay.getvalue(),
                        file_name="oil_spill_overlay.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col_dl3:
                    # Download report
                    report = f"""OIL SPILL DETECTION REPORT
                    
Image Analysis Results:
- Original Image Size: {original_size}
- Spill Coverage: {stats['coverage']:.4f}%
- Oil Pixels: {stats['oil_pixels']:,}
- Total Pixels: {stats['total_pixels']:,}
- Detected Spills: {stats['num_spills']}
- Max Confidence: {stats['max_confidence']:.1f}%
- Mean Confidence: {stats['mean_confidence']:.1f}%

Detection Settings:
- Confidence Threshold: {confidence_threshold}
- Input Size: {input_size}
- Mask Cleaning: {enable_cleaning}

Timestamp: {st.session_state.get('timestamp', 'N/A')}
"""
                    st.download_button(
                        label="📥 Download Report",
                        data=report,
                        file_name="oil_spill_report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("💡 Try adjusting the confidence threshold or input size")

else:
    st.info("👆 **Please upload a satellite image to begin oil spill detection**")
    
    # Show sample usage
    with st.expander("📚 How to use"):
        st.markdown("""
        **For best results:**
        1. **Upload clear satellite imagery** with visible water bodies
        2. **Start with default settings** (threshold: 0.5, size: 256)
        3. **Adjust confidence threshold** if detection is poor:
           - Too many false positives? → Increase threshold (0.7-0.9)
           - Missing real spills? → Decrease threshold (0.3-0.5)
        4. **Try different input sizes** for better detection of small spills
        
        **Expected output:**
        - **Red overlays** show detected oil spills
        - **White masks** show binary detection areas
        - **Detailed statistics** help assess spill severity
        """)

# Add requirements info
with st.sidebar.expander("📋 Requirements"):
    st.markdown("""
    **Required packages:**
    ```txt
    streamlit
    torch
    segmentation-models-pytorch
    torchvision
    pillow
    opencv-python
    numpy
    requests
    ```
    """)
