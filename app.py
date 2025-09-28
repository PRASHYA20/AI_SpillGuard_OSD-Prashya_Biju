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
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        # assume output is (B,1,H,W) with sigmoid activation required
        if output.shape[1] == 1:
            output = torch.sigmoid(output)
            mask = (output > 0.5).float()
        else:
            mask = torch.argmax(output, dim=1).unsqueeze(1)

    mask = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
    mask_img = Image.fromarray(mask)
    return mask_img

# -------------------
# Streamlit UI
# -------------------
st.title("🌊 AI SpillGuard - Oil Spill Segmentation")

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file is not None:
    # Load input image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run prediction
    mask = predict(image)
    st.image(mask, caption="Predicted Oil Spill Mask", use_container_width=True)

    # Overlay
    image_resized = image.resize(mask.size)
    overlay = cv2.addWeighted(
        np.array(image_resized),
        0.7,
        cv2.cvtColor(np.array(mask), cv2.COLOR_GRAY2RGB),
        0.3,
        0
    )
    st.image(overlay, caption="Overlay Result", use_container_width=True)
