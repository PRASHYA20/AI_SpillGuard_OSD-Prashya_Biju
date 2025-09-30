import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import io
import os
import collections
import torchvision.transforms as T
import segmentation_models_pytorch as smp  # ✅ you had this in requirements

# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(page_title="Oil Spill Segmentation", layout="wide")
st.title("🌊 Oil Spill Segmentation")
st.write("Upload a satellite image to detect oil spills.")

# ------------------------------
# Model Loader
# ------------------------------
@st.cache_resource
def load_model(model_path, arch="unet", encoder="resnet50", num_classes=1):
    """Load trained model with correct architecture"""
    try:
        if arch.lower() == "unet":
            model = smp.Unet(
                encoder_name=encoder,        # e.g. "resnet50"
                encoder_weights=None,        # we load our own weights
                in_channels=3,
                classes=num_classes,
            )
        elif arch.lower() == "deeplabv3+":
            model = smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=3,
                classes=num_classes,
            )
        else:
            raise ValueError(f"Unsupported arch: {arch}")

        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, collections.OrderedDict):
            model.load_state_dict(checkpoint, strict=False)
        elif isinstance(checkpoint, nn.Module):
            model = checkpoint
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# ------------------------------
# Upload Section
# ------------------------------
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.01, 0.99, 0.5, 0.01)
min_spill_size = st.sidebar.slider("Minimum Spill Size", 10, 5000, 100, 10)

uploaded_model = st.sidebar.file_uploader("Upload trained model (.pth)", type=["pth"])
arch_choice = st.sidebar.selectbox("Model Architecture", ["unet", "deeplabv3+"])
encoder_choice = st.sidebar.selectbox("Encoder Backbone", ["resnet50", "resnet34", "timm-efficientnet-b0"])

model = None
if uploaded_model is not None:
    with open("uploaded_model.pth", "wb") as f:
        f.write(uploaded_model.getvalue())
    model = load_model("uploaded_model.pth", arch=arch_choice, encoder=encoder_choice)

# ------------------------------
# Image Uploader
# ------------------------------
st.header("📡 Upload Satellite Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# ------------------------------
# Preprocessing
# ------------------------------
def preprocess(image_pil, size=256):
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return transform(image_pil).unsqueeze(0)

# ------------------------------
# Postprocessing (Extract Mask)
# ------------------------------
def postprocess(prob_map, orig_size, threshold=0.5):
    mask = (prob_map > threshold).astype(np.uint8) * 255
    return cv2.resize(mask, orig_size, interpolation=cv2.INTER_NEAREST)

def extract_contours(mask, min_area=100):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        shapes.append(cnt)
    return shapes

# ------------------------------
# Main Logic
# ------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    orig_size = image.size

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🛰️ Original")
        st.image(image, use_container_width=True)

    if model is None:
        st.warning("⚠️ No model loaded. Please upload your `.pth` file.")
    else:
        with st.spinner("🔍 Segmenting oil spill..."):
            inp = preprocess(image, size=256)
            with torch.no_grad():
                out = model(inp)
            if isinstance(out, (list, tuple)):
                out = out[0]
            prob_map = torch.sigmoid(out).squeeze().cpu().numpy()

            mask = postprocess(prob_map, orig_size, threshold=confidence_threshold)
            contours = extract_contours(mask, min_area=min_spill_size)

            # Overlay
            overlay = np.array(image).copy()
            cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
            overlay = cv2.addWeighted(np.array(image), 0.7, overlay, 0.3, 0)

        with col2:
            st.subheader("🎭 Prediction")
            st.image(overlay, use_container_width=True)
            st.write(f"Detected {len(contours)} oil spill region(s).")

        # Downloads
        st.subheader("💾 Download Results")
        mask_pil = Image.fromarray(mask)
        buf = io.BytesIO()
        mask_pil.save(buf, format="PNG")
        st.download_button(
            "Download Predicted Mask",
            data=buf.getvalue(),
            file_name="oil_spill_mask.png",
            mime="image/png"
        )
