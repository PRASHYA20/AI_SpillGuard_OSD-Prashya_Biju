import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import io

# -----------------------------
# Dropbox Model URL
# -----------------------------
MODEL_PATH = "oil_spill_model_deploy.pth"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/stl47n6ixrzv59xs2jt4m/oil_spill_model_deploy.pth?rlkey=rojyk0fq73mk8tai8jc3exrev&st=w6qm08lh&dl=1"

# -----------------------------
# Define your UNet model (example)
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(UNet, self).__init__()
        self.dc1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.dc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.dc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.dc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.dc5 = DoubleConv(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dc6 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dc7 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dc8 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dc9 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        x1 = self.dc1(x)
        x2 = self.dc2(self.pool1(x1))
        x3 = self.dc3(self.pool2(x2))
        x4 = self.dc4(self.pool3(x3))
        x5 = self.dc5(self.pool4(x4))
        x = self.up1(x5)
        x = self.dc6(torch.cat([x, x4], dim=1))
        x = self.up2(x)
        x = self.dc7(torch.cat([x, x3], dim=1))
        x = self.up3(x)
        x = self.dc8(torch.cat([x, x2], dim=1))
        x = self.up4(x)
        x = self.dc9(torch.cat([x, x1], dim=1))
        x = self.out_conv(x)
        return x

# -----------------------------
# Download Model if not exists
# -----------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("🔽 Downloading model from Dropbox...")
        r = requests.get(DROPBOX_URL, allow_redirects=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    download_model()
    model = UNet(in_ch=3, out_ch=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# -----------------------------
# Preprocessing
# -----------------------------
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌊 Oil Spill Segmentation (UNet)")
st.write("Upload a satellite image to detect possible oil spills.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()

    # Threshold
    mask = (pred > 0.5).astype(np.uint8) * 255

    # Display
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(image)
    ax[1].imshow(mask, cmap="Reds", alpha=0.5)
    ax[1].set_title("Predicted Oil Spill Mask")
    ax[1].axis("off")

    st.pyplot(fig)

    # -----------------------------
    # Download Mask Button
    # -----------------------------
    mask_img = Image.fromarray(mask.astype(np.uint8))
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="💾 Download Predicted Mask",
        data=byte_im,
        file_name="oil_spill_mask.png",
        mime="image/png"
    )
