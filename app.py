import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

# -------------------
# Device Setup
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Load Model
# -------------------
MODEL_PATH = "oil_spill_model_deploy.pth"

@st.cache_resource
def load_model():
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    return model

model = load_model()

# -------------------
# Image Transform
# -------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),   # resize to training size
    transforms.ToTensor(),
])

# -------------------
# Prediction Function
# -------------------
def predict(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        if output.shape[1] == 1:  # binary segmentation
            output = torch.sigmoid(output)
            mask = (output > 0.5).float()
        else:  # multi-class segmentation
            mask = torch.argmax(output, dim=1).unsqueeze(1)

    mask = mask.squeeze().cpu().numpy().astype(np.uint8)
    return mask

# -------------------
# Overlay Function
# -------------------
def create_overlay(original: Image.Image, mask: np.ndarray):
    # Resize original image to match mask
    original_resized = original.resize((mask.shape[1], mask.shape[0]))
    original_np = np.array(original_resized)

    # Create a red mask where oil spill = 1
    color_mask = np.zeros_like(original_np)
    color_mask[mask == 1] = [255, 0, 0]  # Red for oil spill

    # Overlay with transparency
    overlay = cv2.addWeighted(original_np, 0.7, color_mask, 0.3, 0)
    return overlay

# -------------------
# Streamlit UI
# -------------------
st.title("🌊 AI SpillGuard - Oil Spill Segmentation with Overlay")

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file is not None:
    # Load input image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run prediction
    mask = predict(image)

    # Show raw mask
    st.image(mask * 255, caption="Predicted Oil Spill Mask", use_container_width=True)

    # Show overlay
    overlay = create_overlay(image, mask)
    st.image(overlay, caption="Overlay Result (Oil Spill in Red)", use_container_width=True)
