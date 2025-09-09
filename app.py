import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np

# Set page layout
st.set_page_config(page_title="Oil Spill Segmentation", layout="wide")

# -------- Load Model --------
@st.cache_resource
def load_model():
    # Create model with auxiliary classifier enabled to match the saved state
    model = deeplabv3_resnet50(pretrained=False, aux_loss=True)
    
    # Modify the main classifier for binary segmentation (1 output channel)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1))
    
    # Modify auxiliary classifier if it exists
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1))
    
    # Load the saved weights from the correct file path
    model.load_state_dict(torch.load(r"C:\Users\prash\OneDrive\ai_project\deeplabv3_oilspill(1).pth", map_location='cpu'))
    
    # Set model to evaluation mode
    model.eval()
    return model

# Load model once and cache it
model = load_model()

# -------- Preprocess Image --------
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)

# -------- Postprocess Output --------
def postprocess_output(output_tensor):
    output = torch.sigmoid(output_tensor).detach().cpu().numpy()[0, 0]
    output = (output > 0.5).astype(np.uint8) * 255
    return output

# -------- Streamlit Interface --------
st.title("Oil Spill Segmentation System")
st.write("Upload a satellite image and view oil spill areas detected by the AI model.")

# File upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting oil spill..."):
        # Preprocess image
        input_tensor = preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)['out']
        
        # Postprocess the result
        mask = postprocess_output(output)

    # Display result
    st.image(mask, caption="Detected Oil Spill", use_column_width=True)
    st.success("Segmentation completed!")

    # Optional: Save result
    save_option = st.checkbox("Save the result image")
    if save_option:
        output_image = Image.fromarray(mask)
        output_image.save("result.png")
        st.write("Image saved as result.png")
