import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import io
import os

# Set page config
st.set_page_config(page_title="Oil Spill Segmentation", layout="wide")

st.title("🛢️ Oil Spill Segmentation Model")
st.write("Upload satellite imagery to detect oil spills")

# Define the model architecture to match your trained model
class ResNetUNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNetUNet, self).__init__()
        
        # Encoder - ResNet (matches your checkpoint structure)
        self.encoder = models.resnet50(pretrained=False)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)
        
        # Decoder
        x = self.up1(x4)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = self.conv3(x)
        
        x = self.final(x)
        return torch.sigmoid(x)

# Alternative simpler architecture that matches your checkpoint
class OilSpillModel(nn.Module):
    def __init__(self, num_classes=1):
        super(OilSpillModel, self).__init__()
        
        # Use ResNet50 as encoder
        resnet = models.resnet50(pretrained=False)
        
        # Encoder layers
        self.encoder_conv1 = resnet.conv1
        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool
        
        self.encoder_layer1 = resnet.layer1
        self.encoder_layer2 = resnet.layer2
        self.encoder_layer3 = resnet.layer3
        self.encoder_layer4 = resnet.layer4
        
        # Decoder (simplified)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encoder
        x = self.encoder_conv1(x)
        x = self.encoder_bn1(x)
        x = self.encoder_relu(x)
        x = self.encoder_maxpool(x)

        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        
        # Decoder
        x = self.decoder(x)
        return x

@st.cache_resource
def load_oil_spill_model():
    """Load your specific oil spill model"""
    try:
        # Initialize model
        model = OilSpillModel(num_classes=1)
        
        # Load your specific checkpoint
        checkpoint = torch.load('oil_spill_model_deploy.pth', map_location='cpu')
        
        # Load state dict
        model.load_state_dict(checkpoint)
        
        model.eval()
        st.sidebar.success("✅ Oil Spill Model Loaded!")
        return model
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None

# Load model
model = load_oil_spill_model()

# Model info
st.sidebar.header("Model Information")
if model is not None:
    st.sidebar.success("✅ ResNet50-based Segmentation Model")
    total_params = sum(p.numel() for p in model.parameters())
    st.sidebar.write(f"**Parameters:** {total_params:,}")
else:
    st.sidebar.error("❌ Model not loaded")

# Image preprocessing
def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for model input"""
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    original_size = image.size
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to numpy and normalize
    img_array = np.array(image) / 255.0
    
    # Normalize with ImageNet stats (common for ResNet)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC to BCHW
    
    return img_tensor, original_size

def postprocess_mask(mask, original_size, threshold=0.5):
    """Postprocess model output mask"""
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().cpu().numpy()
    
    # Apply threshold
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    
    # Resize to original
    mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    return mask_resized, mask

def create_oil_spill_visualization(original_image, mask, alpha=0.6):
    """Create oil spill visualization"""
    original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    
    # Create colored mask (red for oil spills)
    colored_mask = np.zeros_like(original_cv)
    colored_mask[mask > 0] = [0, 0, 255]  # Red color for oil spills
    
    # Blend
    blended = cv2.addWeighted(original_cv, 1 - alpha, colored_mask, alpha, 0)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    
    return blended_rgb

# Settings
st.sidebar.header("⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Detection Threshold", 
    min_value=0.1, 
    max_value=0.9, 
    value=0.5, 
    step=0.1,
    help="Higher values = more conservative detection"
)

# Main app
if model is not None:
    st.header("📡 Upload Satellite Imagery")
    
    uploaded_file = st.file_uploader(
        "Choose satellite image", 
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Upload satellite imagery for oil spill detection"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        original_size = image.size
        
        # Display layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🛰️ Original Image")
            st.image(image, use_column_width=True)
            st.write(f"Size: {original_size}")
        
        # Process and predict
        with st.spinner("🔍 Analyzing for oil spills..."):
            try:
                # Preprocess
                input_tensor, original_size = preprocess_image(image)
                
                # Prediction
                with torch.no_grad():
                    output = model(input_tensor)
                    probability_mask = output.squeeze().cpu().numpy()
                
                # Postprocess
                binary_mask, prob_map = postprocess_mask(
                    probability_mask, 
                    original_size, 
                    confidence_threshold
                )
                
                # Display results
                with col2:
                    st.subheader("🎭 Detection Mask")
                    st.image(binary_mask, use_column_width=True, clamp=True)
                    
                    # Probability map
                    if st.checkbox("Show Confidence Map"):
                        prob_display = (prob_map * 255).astype(np.uint8)
                        prob_colored = cv2.applyColorMap(prob_display, cv2.COLORMAP_JET)
                        prob_resized = cv2.resize(prob_colored, original_size)
                        st.image(prob_resized, use_column_width=True)
                
                with col3:
                    st.subheader("🛢️ Oil Spill Overlay")
                    overlay = create_oil_spill_visualization(image, binary_mask)
                    st.image(overlay, use_column_width=True)
                
                # Oil spill statistics
                st.subheader("📊 Oil Spill Analysis")
                
                total_pixels = binary_mask.size
                oil_pixels = np.sum(binary_mask > 0)
                oil_percentage = (oil_pixels / total_pixels) * 100
                
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                
                with col_stats1:
                    st.metric("Oil Spill Area (pixels)", f"{oil_pixels:,}")
                with col_stats2:
                    st.metric("Coverage", f"{oil_percentage:.2f}%")
                with col_stats3:
                    st.metric("Max Confidence", f"{np.max(probability_mask):.3f}")
                with col_stats4:
                    st.metric("Detection Quality", 
                             "High" if oil_percentage > 1 else "Low" if oil_pixels > 0 else "None")
                
                # Alert system
                if oil_pixels > 0:
                    st.warning(f"🚨 Oil spill detected! {oil_percentage:.2f}% of area affected")
                else:
                    st.success("✅ No oil spills detected")
                
                # Download section
                st.subheader("💾 Download Results")
                
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                
                with col_dl1:
                    # Binary mask
                    mask_pil = Image.fromarray(binary_mask)
                    buf_mask = io.BytesIO()
                    mask_pil.save(buf_mask, format='PNG')
                    st.download_button(
                        label="Download Detection Mask",
                        data=buf_mask.getvalue(),
                        file_name="oil_spill_mask.png",
                        mime="image/png"
                    )
                
                with col_dl2:
                    # Overlay
                    overlay_pil = Image.fromarray(overlay)
                    buf_overlay = io.BytesIO()
                    overlay_pil.save(buf_overlay, format='PNG')
                    st.download_button(
                        label="Download Overlay",
                        data=buf_overlay.getvalue(),
                        file_name="oil_spill_overlay.png",
                        mime="image/png"
                    )
                
                with col_dl3:
                    # Analysis report
                    report = f"""
                    OIL SPILL DETECTION REPORT
                    =========================
                    Image Size: {original_size}
                    Total Pixels: {total_pixels:,}
                    Oil Spill Pixels: {oil_pixels:,}
                    Area Coverage: {oil_percentage:.2f}%
                    Detection Threshold: {confidence_threshold}
                    Max Confidence: {np.max(probability_mask):.3f}
                    
                    Timestamp: {st.session_state.get('timestamp', 'N/A')}
                    """
                    
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name="oil_spill_report.txt",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("💡 This might be an architecture mismatch. Trying alternative approach...")

else:
    st.error("""
    ❌ Model failed to load!
    
    **Quick fix:** Rename your model file:
    ```bash
    # Make sure the filename matches exactly
    mv oil_spill_model_deploy.pth oil_spill_model_deploy.pth
    ```
    
    **Or update the app to use your exact filename**
    """)

# Model architecture info
with st.expander("🔧 Model Architecture Info"):
    st.markdown("""
    **Your Model Details:**
    - **Base Architecture:** ResNet50 Encoder
    - **Task:** Semantic Segmentation (Oil Spill Detection)
    - **Output:** Binary mask (Oil vs No Oil)
    - **Activation:** Sigmoid (for binary classification)
    
    **Model File:** `oil_spill_model_deploy.pth`
    """)

# Instructions
with st.expander("📚 How to Use"):
    st.markdown("""
    1. **Upload** satellite imagery in common formats (PNG, JPG, etc.)
    2. **Adjust** the detection threshold based on your needs
    3. **View** the detection results in three panels
    4. **Download** masks, overlays, and analysis reports
    5. **Monitor** the oil spill statistics and alerts
    
    **Tip:** Higher thresholds reduce false positives but might miss smaller spills.
    """)
