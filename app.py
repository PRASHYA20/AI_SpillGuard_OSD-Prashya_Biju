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

def test_all_strategies(model, image, device, confidence_threshold):
    """Test all strategies side by side"""
    st.subheader("🔍 Testing All Detection Strategies")
    
    strategies = []
    
    # Original
    input_tensor, processed_img = preprocess_image(image)
    input_tensor = input_tensor.to(device, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
    strategies.append(("Standard", pred, processed_img))
    
    # Enhanced versions
    enhancements = [
        ("High Contrast", lambda img: ImageEnhance.Contrast(img).enhance(2.0)),
        ("Sharpened", lambda img: img.filter(ImageFilter.SHARPEN)),
        ("Brightened", lambda img: ImageEnhance.Brightness(img).enhance(1.4)),
        ("Color Enhanced", lambda img: ImageEnhance.Color(img).enhance(1.5))
    ]
    
    for name, enhance_func in enhancements:
        enhanced_img = enhance_func(image)
        input_tensor, processed_img = preprocess_image(enhanced_img)
        input_tensor = input_tensor.to(device, dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
        strategies.append((name, pred, processed_img))
    
    # Display all strategies
    cols = st.columns(3)
    for idx, (name, pred, proc_img) in enumerate(strategies):
        with cols[idx % 3]:
            binary_mask = (pred > confidence_threshold).astype(np.uint8) * 255
            overlay = proc_img.copy()
            overlay_np = np.array(overlay)
            overlay_np[binary_mask>0] = [255,0,0]
            overlay_img = Image.fromarray(overlay_np)
            
            st.image(overlay_img, caption=f"{name}", use_column_width=True)
            spill_area = np.sum(binary_mask>0) / (binary_mask.shape[0]*binary_mask.shape[1]) * 100
            st.write(f"Spill Area: {spill_area:.2f}%")

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊", layout="wide")
st.title("🌊 Oil Spill Segmentation with UNet (ResNet34)")
st.write("Upload a satellite image to detect oil spills.")

with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.01, 0.9, 0.3, 0.01)
    
    st.header("🎛️ Detection Mode")
    detection_mode = st.radio(
        "Choose detection mode:",
        ["Standard", "Enhanced", "Test All Strategies"]
    )
    
    st.header("ℹ️ About")
    st.write("This app uses a UNet model (ResNet34 backbone) for oil spill segmentation.")
    st.write("**Enhanced Mode** helps with difficult images that standard detection misses.")

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
        if detection_mode == "Test All Strategies":
            # Test all strategies side by side
            test_all_strategies(
                st.session_state.model, 
                image, 
                st.session_state.device,
                confidence_threshold
            )
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

                # Binary mask
                binary_mask = (prediction > confidence_threshold).astype(np.uint8) * 255

                # Overlay mask on original
                overlay = processed_image.copy()
                overlay_np = np.array(overlay)
                overlay_np[binary_mask>0] = [255,0,0]  # red for oil spill
                overlay_img = Image.fromarray(overlay_np)

                col1, col2 = st.columns(2)
                col1.image(processed_image, caption="Processed Image", use_column_width=True)
                col2.image(overlay_img, caption="Oil Spill Overlay", use_column_width=True)

                # Metrics
                spill_area = np.sum(binary_mask>0) / (binary_mask.shape[0]*binary_mask.shape[1]) * 100
                max_conf = np.max(prediction) * 100
                mean_conf = np.mean(prediction) * 100

                st.subheader("📊 Detection Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Spill Area", f"{spill_area:.2f}%")
                col2.metric("Max Confidence", f"{max_conf:.1f}%")
                col3.metric("Mean Confidence", f"{mean_conf:.1f}%")
                
                status = "🔴 Spill Detected" if spill_area > 0.1 else "🟢 No Spill"
                st.metric("Status", status)

                # Download mask
                mask_image = Image.fromarray(binary_mask)
                buf = io.BytesIO()
                mask_image.save(buf, format="PNG")
                st.download_button(
                    label="💾 Download Prediction Mask",
                    data=buf.getvalue(),
                    file_name="oil_spill_mask.png",
                    mime="image/png"
                )

                # Debug info for difficult images
                if spill_area < 1.0 and detection_mode == "Standard":
                    st.info("💡 **Tip**: Try 'Enhanced' or 'Test All Strategies' mode if you expected more detection!")
else:
    st.info("👆 Please upload a satellite image to begin detection.")
