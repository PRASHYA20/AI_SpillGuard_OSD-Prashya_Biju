import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import requests
import os
import io

# -----------------------------
# Dropbox Model URL
# -----------------------------
MODEL_PATH = "oil_spill_model_deploy.pth"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/stl47n6ixrzv59xs2jt4m/oil_spill_model_deploy.pth?rlkey=rojyk0fq73mk8tai8jc3exrev&dl=1"

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
# Create UNet Model (using segmentation_models_pytorch)
# -----------------------------
def create_unet_model():
    try:
        # Create the same model architecture that was used for training
        model = smp.Unet(
            encoder_name="resnet34",        # This matches your trained model
            encoder_weights=None,           # Don't load ImageNet weights
            in_channels=3,
            classes=1,
            activation=None,
        )
        return model
    except Exception as e:
        st.error(f"❌ Error creating model: {e}")
        return None

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write(f"🖥️ Using device: {device}")

        if not download_model():
            return None, device

        # Create model with correct architecture
        model = create_unet_model()
        if model is None:
            return None, device

        # Load the pre-trained weights
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Remove DataParallel prefix if exists
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        # Load the state dict
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
    try:
        # Resize image to 256x256 (common size for segmentation models)
        image_resized = image.resize((256, 256))
        img_array = np.array(image_resized).astype(np.float32) / 255.0

        # ImageNet normalization (same as during training)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std

        # Convert to tensor: [C, H, W] and add batch dimension
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
        return img_tensor, image_resized
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {e}")
        return None, None

# -----------------------------
# Postprocess Prediction
# -----------------------------
def postprocess_prediction(prediction, original_size, confidence_threshold=0.5):
    try:
        # Convert prediction to numpy
        pred_mask = prediction.squeeze().cpu().numpy()
        
        # Apply confidence threshold
        binary_mask = (pred_mask > confidence_threshold).astype(np.uint8) * 255
        
        # Resize mask back to original image size
        mask_image = Image.fromarray(binary_mask)
        mask_resized = mask_image.resize(original_size, Image.NEAREST)
        
        return mask_resized, pred_mask
    except Exception as e:
        st.error(f"❌ Error postprocessing prediction: {e}")
        return None, None

# -----------------------------
# Create Overlay
# -----------------------------
def create_overlay(original_image, mask):
    try:
        # Convert images to numpy arrays
        original_np = np.array(original_image)
        mask_np = np.array(mask)
        
        # Create overlay (red for oil spills)
        overlay = original_np.copy()
        overlay[mask_np > 0] = [255, 0, 0]  # Red color for oil spills
        
        # Blend overlay with original (50% transparency)
        alpha = 0.5
        blended = (original_np * (1 - alpha) + overlay * alpha).astype(np.uint8)
        
        return Image.fromarray(blended)
    except Exception as e:
        st.error(f"❌ Error creating overlay: {e}")
        return original_image

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊", layout="wide")
st.title("🌊 Oil Spill Segmentation with AI")
st.write("Upload a satellite image to detect oil spills using deep learning.")

with st.sidebar:
    st.header("⚙️ Detection Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05,
                                   help="Higher values = more confident detections")
    st.header("ℹ️ About")
    st.write("This app uses a UNet deep learning model with ResNet34 encoder to detect oil spills in satellite imagery.")

# Initialize model
if 'model' not in st.session_state:
    with st.spinner("🔄 Loading AI model..."):
        model, device = load_model()
        st.session_state.model = model
        st.session_state.device = device

# Upload image
uploaded_file = st.file_uploader("📤 Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load original image
        original_image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(original_image, use_column_width=True)
        
        if st.session_state.model is None:
            st.error("❌ Model failed to load. Please check the model file.")
        else:
            if st.button("🎯 Detect Oil Spills", type="primary"):
                with st.spinner("🔄 Analyzing image for oil spills..."):
                    # Preprocess image
                    input_tensor, processed_image = preprocess_image(original_image)
                    
                    if input_tensor is not None:
                        # Move to device and predict
                        input_tensor = input_tensor.to(st.session_state.device)
                        
                        with torch.no_grad():
                            output = st.session_state.model(input_tensor)
                            prediction = torch.sigmoid(output)
                        
                        # Postprocess prediction
                        pred_mask, confidence_map = postprocess_prediction(
                            prediction, original_image.size, confidence_threshold
                        )
                        
                        if pred_mask is not None:
                            # Create overlay
                            overlay_image = create_overlay(original_image, pred_mask)
                            
                            with col2:
                                st.subheader("🔍 Detection Results")
                                
                                # Show overlay
                                st.image(overlay_image, 
                                       caption="Oil Spill Detection (Red = Oil Spill)", 
                                       use_column_width=True)
                                
                                # Show prediction mask
                                st.image(pred_mask, 
                                       caption="Prediction Mask", 
                                       use_column_width=True,
                                       clamp=True)
                            
                            # Calculate metrics
                            mask_array = np.array(pred_mask)
                            spill_pixels = np.sum(mask_array > 0)
                            total_pixels = mask_array.size
                            spill_percentage = (spill_pixels / total_pixels) * 100
                            max_confidence = np.max(confidence_map) * 100
                            
                            # Display metrics
                            st.subheader("📊 Detection Metrics")
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            
                            with metric_col1:
                                st.metric("Spill Area", f"{spill_percentage:.2f}%")
                            
                            with metric_col2:
                                st.metric("Max Confidence", f"{max_confidence:.1f}%")
                            
                            with metric_col3:
                                status = "🔴 Spill Detected" if spill_percentage > 0.1 else "🟢 No Spill"
                                st.metric("Status", status)
                            
                            # Download buttons
                            st.subheader("💾 Download Results")
                            download_col1, download_col2 = st.columns(2)
                            
                            with download_col1:
                                # Download mask
                                mask_buffer = io.BytesIO()
                                pred_mask.save(mask_buffer, format="PNG")
                                st.download_button(
                                    label="Download Prediction Mask",
                                    data=mask_buffer.getvalue(),
                                    file_name="oil_spill_mask.png",
                                    mime="image/png"
                                )
                            
                            with download_col2:
                                # Download overlay
                                overlay_buffer = io.BytesIO()
                                overlay_image.save(overlay_buffer, format="PNG")
                                st.download_button(
                                    label="Download Overlay Image",
                                    data=overlay_buffer.getvalue(),
                                    file_name="oil_spill_overlay.png",
                                    mime="image/png"
                                )
                            
                            st.balloons()
                            
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

else:
    st.info("👆 Please upload a satellite image to begin oil spill detection")

# Model information
with st.expander("🔧 Model Information"):
    st.write("""
    **Model Architecture:** UNet with ResNet34 encoder
    **Input Size:** 256x256 pixels  
    **Output:** Binary segmentation mask
    **Training:** Trained on satellite imagery with oil spill annotations
    """)
