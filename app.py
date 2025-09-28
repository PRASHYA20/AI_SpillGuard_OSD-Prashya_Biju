import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import io
import requests
import os

# -----------------------------
# Model Configuration
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
# Download Model
# -----------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model...")
        try:
            response = requests.get(DROPBOX_URL, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with open(MODEL_PATH, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(progress)
                            status_text.text(f"Downloaded: {downloaded/(1024*1024):.1f}MB")
            
            progress_bar.empty()
            status_text.empty()
            st.success("✅ Model downloaded successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Download failed: {e}")
            return False
    return True

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"🔧 Using device: {device}")
    
    if not download_model():
        return None, device

    try:
        model = create_unet_model()
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint)
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "") if key.startswith("module.") else key
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        
        st.success("✅ Model loaded successfully!")
        return model, device
        
    except Exception as e:
        st.error(f"❌ Model loading error: {e}")
        return None, device

# -----------------------------
# Preprocess Image
# -----------------------------
def preprocess_image(image):
    # Resize to model input size
    image_resized = image.resize((256, 256))
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
    
    return img_tensor, image_resized

# -----------------------------
# Post-process Mask
# -----------------------------
def postprocess_mask(prediction, confidence_threshold):
    # Apply sigmoid and threshold
    prob_mask = torch.sigmoid(prediction).squeeze().cpu().numpy()
    binary_mask = (prob_mask > confidence_threshold).astype(np.uint8) * 255
    return binary_mask, prob_mask

# -----------------------------
# Create Visualization
# -----------------------------
def create_visualization(original_image, binary_mask):
    # Create overlay (red for oil spills)
    overlay = original_image.copy()
    overlay_np = np.array(overlay)
    
    # Apply red color to detected oil areas
    overlay_np[binary_mask > 0] = [255, 0, 0]  # Red color
    
    return Image.fromarray(overlay_np)

# -----------------------------
# Calculate Statistics
# -----------------------------
def calculate_statistics(binary_mask, prob_map):
    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
    oil_pixels = np.sum(binary_mask > 0)
    spill_area_percent = (oil_pixels / total_pixels) * 100
    max_confidence = np.max(prob_map) * 100
    
    return {
        'spill_area_percent': spill_area_percent,
        'oil_pixels': oil_pixels,
        'total_pixels': total_pixels,
        'max_confidence': max_confidence
    }

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Oil Spill Detection System")
st.markdown("Upload a satellite or aerial image to detect oil spills using AI.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Higher values = more conservative detection"
    )
    
    st.header("📊 Model Info")
    if 'model' not in st.session_state:
        with st.spinner("Loading model..."):
            model, device = load_model()
            st.session_state.model = model
            st.session_state.device = device
    else:
        st.success("✅ Model ready!")

# Main content
uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"],
    help="Upload satellite or aerial imagery"
)

if uploaded_file is not None:
    # Load and display original image
    original_image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(original_image, use_column_width=True)
    
    if st.session_state.model is None:
        st.error("❌ Model failed to load. Please check the console for errors.")
    else:
        if st.button("🔍 Detect Oil Spills", type="primary"):
            with st.spinner("Analyzing image..."):
                # Preprocess
                input_tensor, processed_image = preprocess_image(original_image)
                input_tensor = input_tensor.to(st.session_state.device)
                
                # Prediction
                with torch.no_grad():
                    output = st.session_state.model(input_tensor)
                    binary_mask, prob_map = postprocess_mask(output, confidence_threshold)
                
                # Create visualization
                result_image = create_visualization(processed_image, binary_mask)
                
                # Calculate statistics
                stats = calculate_statistics(binary_mask, prob_map)
                
                # Display results
                with col2:
                    st.subheader("🛢️ Detection Results")
                    st.image(result_image, use_column_width=True)
                    
                    # Metrics
                    st.metric("Oil Spill Coverage", f"{stats['spill_area_percent']:.2f}%")
                    st.metric("Confidence Level", f"{stats['max_confidence']:.1f}%")
                    
                    # Status
                    if stats['spill_area_percent'] > 1.0:
                        st.error("🚨 Oil spill detected!")
                    else:
                        st.success("✅ No significant oil spill detected")
                
                # Download section
                st.subheader("💾 Download Results")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    # Download mask
                    mask_image = Image.fromarray(binary_mask)
                    buf_mask = io.BytesIO()
                    mask_image.save(buf_mask, format="PNG")
                    st.download_button(
                        label="Download Binary Mask",
                        data=buf_mask.getvalue(),
                        file_name="oil_spill_mask.png",
                        mime="image/png"
                    )
                
                with col4:
                    # Download overlay
                    buf_overlay = io.BytesIO()
                    result_image.save(buf_overlay, format="PNG")
                    st.download_button(
                        label="Download Overlay Image",
                        data=buf_overlay.getvalue(),
                        file_name="oil_spill_overlay.png",
                        mime="image/png"
                    )
                
                # Detailed statistics
                with st.expander("📈 Detailed Analysis"):
                    st.write(f"**Total Pixels:** {stats['total_pixels']:,}")
                    st.write(f"**Oil Pixels:** {stats['oil_pixels']:,}")
                    st.write(f"**Spill Area Ratio:** {stats['spill_area_percent']:.4f}%")
                    st.write(f"**Max Confidence:** {stats['max_confidence']:.2f}%")

else:
    st.info("👆 Please upload an image to start oil spill detection.")
    st.markdown("""
    ### 💡 Tips for best results:
    - Use clear satellite or aerial images
    - Ensure good lighting conditions
    - Adjust confidence threshold if needed
    - Images with water bodies work best
    """)
