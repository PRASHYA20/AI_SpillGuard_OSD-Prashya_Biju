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

def adaptive_threshold_detection(prediction, method="otsu"):
    """Use adaptive thresholding instead of fixed threshold"""
    from skimage import filters
    
    if method == "otsu":
        threshold = filters.threshold_otsu(prediction)
    elif method == "mean":
        threshold = np.mean(prediction)
    elif method == "median":
        threshold = np.median(prediction)
    else:
        threshold = np.percentile(prediction, 75)  # Top 25%
    
    binary_mask = (prediction > threshold).astype(np.uint8) * 255
    return binary_mask, threshold

def test_all_strategies_with_adaptive_threshold(model, image, device, base_confidence):
    """Test all strategies with adaptive thresholding"""
    st.subheader("🔍 Testing All Strategies with Adaptive Thresholding")
    
    strategies = []
    
    # Test different preprocessing strategies
    enhancements = [
        ("Standard", lambda img: img),
        ("High Contrast", lambda img: ImageEnhance.Contrast(img).enhance(2.5)),
        ("Very High Contrast", lambda img: ImageEnhance.Contrast(img).enhance(3.5)),
        ("Sharpened", lambda img: img.filter(ImageFilter.SHARPEN)),
        ("Double Sharpened", lambda img: img.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)),
        ("Brightened", lambda img: ImageEnhance.Brightness(img).enhance(1.6)),
        ("Color Enhanced", lambda img: ImageEnhance.Color(img).enhance(2.0)),
        ("Edge Enhance", lambda img: img.filter(ImageFilter.EDGE_ENHANCE_MORE)),
    ]
    
    for name, enhance_func in enhancements:
        enhanced_img = enhance_func(image)
        input_tensor, processed_img = preprocess_image(enhanced_img)
        input_tensor = input_tensor.to(device, dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
        strategies.append((name, pred, processed_img, enhanced_img))
    
    # Display all strategies with multiple thresholding methods
    st.write("### Fixed Threshold vs Adaptive Threshold")
    
    for strategy_name, prediction, processed_img, enhanced_img in strategies:
        st.write(f"---")
        st.write(f"#### 🔧 {strategy_name}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.image(enhanced_img, caption=f"Enhanced Image", use_column_width=True)
        
        # Fixed low threshold
        with col2:
            low_thresh = base_confidence * 0.3  # Very low threshold
            binary_mask = (prediction > low_thresh).astype(np.uint8) * 255
            overlay = processed_img.copy()
            overlay_np = np.array(overlay)
            overlay_np[binary_mask>0] = [255,0,0]
            overlay_img = Image.fromarray(overlay_np)
            st.image(overlay_img, caption=f"Low Thresh ({low_thresh:.3f})", use_column_width=True)
            spill_area = np.sum(binary_mask>0) / binary_mask.size * 100
            st.write(f"Area: {spill_area:.2f}%")
        
        # Fixed medium threshold
        with col3:
            binary_mask = (prediction > base_confidence).astype(np.uint8) * 255
            overlay = processed_img.copy()
            overlay_np = np.array(overlay)
            overlay_np[binary_mask>0] = [255,0,0]
            overlay_img = Image.fromarray(overlay_np)
            st.image(overlay_img, caption=f"Med Thresh ({base_confidence:.3f})", use_column_width=True)
            spill_area = np.sum(binary_mask>0) / binary_mask.size * 100
            st.write(f"Area: {spill_area:.2f}%")
        
        # Adaptive threshold
        with col4:
            binary_mask, auto_thresh = adaptive_threshold_detection(prediction, "otsu")
            overlay = processed_img.copy()
            overlay_np = np.array(overlay)
            overlay_np[binary_mask>0] = [255,0,0]
            overlay_img = Image.fromarray(overlay_np)
            st.image(overlay_img, caption=f"Auto Thresh ({auto_thresh:.3f})", use_column_width=True)
            spill_area = np.sum(binary_mask>0) / binary_mask.size * 100
            st.write(f"Area: {spill_area:.2f}%")
        
        # Show prediction statistics
        st.write(f"**Prediction Stats:** Min: {prediction.min():.4f}, Max: {prediction.max():.4f}, Mean: {prediction.mean():.4f}")

def debug_prediction_heatmap(prediction, processed_image):
    """Show prediction heatmap to understand what the model sees"""
    st.subheader("🎨 Prediction Heatmap Analysis")
    
    # Normalize prediction for visualization
    pred_normalized = (prediction - prediction.min()) / (prediction.max() - prediction.min())
    
    # Create heatmap
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    ax1.imshow(processed_image)
    ax1.set_title('Processed Image')
    ax1.axis('off')
    
    # Prediction heatmap
    im = ax2.imshow(prediction, cmap='hot')
    ax2.set_title('Model Confidence Heatmap')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2)
    
    # Thresholded areas
    threshold = np.percentile(prediction, 80)  # Show top 20%
    binary_vis = (prediction > threshold).astype(np.float32)
    ax3.imshow(binary_vis, cmap='cool')
    ax3.set_title(f'Top 20% Confidence (>{threshold:.3f})')
    ax3.axis('off')
    
    st.pyplot(fig)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊", layout="wide")
st.title("🌊 Advanced Oil Spill Detection")
st.write("Upload a satellite image to detect oil spills. Use advanced modes for difficult images.")

with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.01, 0.9, 0.1, 0.01)
    
    st.header("🎛️ Detection Mode")
    detection_mode = st.radio(
        "Choose detection mode:",
        ["Standard", "Enhanced", "Advanced Testing", "Debug Analysis"]
    )
    
    st.header("ℹ️ Tips")
    st.write("• Use **Advanced Testing** for difficult images")
    st.write("• Try **very low thresholds** (0.01-0.1) for faint spills")
    st.write("• **Debug Analysis** shows what the model 'sees'")

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
        if detection_mode == "Advanced Testing":
            # Test all strategies with adaptive thresholding
            test_all_strategies_with_adaptive_threshold(
                st.session_state.model, 
                image, 
                st.session_state.device,
                confidence_threshold
            )
        elif detection_mode == "Debug Analysis":
            # Show detailed analysis
            input_tensor, processed_image = preprocess_image(image)
            input_tensor = input_tensor.to(st.session_state.device, dtype=torch.float32)
            with torch.no_grad():
                output = st.session_state.model(input_tensor)
                prediction = torch.sigmoid(output).squeeze().cpu().numpy()
            
            debug_prediction_heatmap(prediction, processed_image)
            
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
                overlay_np[binary_mask>0] = [255,0,0]
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

else:
    st.info("👆 Please upload a satellite image to begin detection.")
