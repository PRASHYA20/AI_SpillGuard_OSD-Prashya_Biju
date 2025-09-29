import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import io
import os

# Set page config
st.set_page_config(page_title="U-Net Mask Prediction", layout="wide")

st.title("🧠 U-Net Mask Prediction")
st.write("Upload an image to generate segmentation masks using U-Net")

# U-Net model definition (adjust based on your architecture)
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super(UNet, self).__init__()
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = self._block(features * 8, features * 16)
        
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)
        
        self.final = nn.Conv2d(features, out_channels, kernel_size=1)
    
    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.final(dec1))

# Enhanced model loading for U-Net
@st.cache_resource
def load_unet_model():
    st.sidebar.write("🔄 Loading U-Net model...")
    
    # Common U-Net model file names
    model_files = [
        'unet_model.pth', 'unet_model.pt', 'model.pth', 'model.pt',
        'unet.pth', 'segmentation_model.pth', 'best_model.pth'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                # Initialize model (adjust parameters based on your model)
                model = UNet(in_channels=3, out_channels=1)
                
                # Load state dict
                if torch.cuda.is_available():
                    checkpoint = torch.load(model_file, map_location='cuda')
                else:
                    checkpoint = torch.load(model_file, map_location='cpu')
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                st.sidebar.success(f"✅ U-Net loaded: {model_file}")
                return model
                
            except Exception as e:
                st.sidebar.warning(f"⚠️ Failed to load {model_file}: {str(e)}")
                continue
    
    st.sidebar.error("❌ No U-Net model found")
    return None

# Load model
model = load_unet_model()

# Display model info
st.sidebar.header("Model Information")
if model is not None:
    st.sidebar.success("✅ U-Net Model Loaded")
    total_params = sum(p.numel() for p in model.parameters())
    st.sidebar.write(f"**Parameters:** {total_params:,}")
else:
    st.sidebar.error("❌ U-Net not loaded")

# Image preprocessing for U-Net
def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for U-Net input"""
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Store original size
    original_size = image.size
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to numpy and normalize
    img_array = np.array(image) / 255.0
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC to BCHW
    
    return img_tensor, original_size

def postprocess_mask(mask, original_size, threshold=0.5):
    """Postprocess U-Net output mask"""
    # Convert to numpy and squeeze
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().cpu().numpy()
    
    # Apply threshold for binary mask
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    
    # Resize to original image size
    mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    return mask_resized, mask  # Return both binary and probability mask

def apply_colormap(mask):
    """Apply colormap to mask"""
    colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    return colored_mask

def create_overlay(original_image, mask, alpha=0.6):
    """Create overlay of original image and mask"""
    # Convert PIL to OpenCV
    original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    
    # Ensure mask is 3-channel
    if len(mask.shape) == 2:
        mask_colored = apply_colormap(mask)
    else:
        mask_colored = mask
    
    # Resize mask to match original
    mask_resized = cv2.resize(mask_colored, (original_cv.shape[1], original_cv.shape[0]))
    
    # Blend
    blended = cv2.addWeighted(original_cv, 1 - alpha, mask_resized, alpha, 0)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    
    return blended_rgb

# Confidence threshold slider
st.sidebar.header("⚙️ Prediction Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=0.9, 
    value=0.5, 
    step=0.1,
    help="Adjust the threshold for binary mask creation"
)

# Main application
if model is not None:
    st.header("📤 Upload Image for Segmentation")
    
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Upload an image for U-Net segmentation"
    )
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        original_size = image.size
        
        # Display in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🖼️ Original Image")
            st.image(image, use_column_width=True)
            st.write(f"Size: {original_size}")
        
        # Process and predict
        with st.spinner("🔄 U-Net is segmenting your image..."):
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
                    st.subheader("🎭 Predicted Mask")
                    st.image(binary_mask, use_column_width=True, clamp=True)
                    
                    # Show probability map
                    if st.checkbox("Show Probability Heatmap"):
                        prob_display = (prob_map * 255).astype(np.uint8)
                        colored_prob = apply_colormap(prob_display)
                        colored_prob_resized = cv2.resize(
                            colored_prob, 
                            (binary_mask.shape[1], binary_mask.shape[0])
                        )
                        st.image(colored_prob_resized, use_column_width=True)
                
                with col3:
                    st.subheader("🎨 Overlay")
                    overlay = create_overlay(image, binary_mask)
                    st.image(overlay, use_column_width=True)
                
                # Statistics
                st.subheader("📊 Segmentation Statistics")
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                
                total_pixels = binary_mask.size
                mask_pixels = np.sum(binary_mask > 0)
                mask_percentage = (mask_pixels / total_pixels) * 100
                
                with col_stats1:
                    st.metric("Mask Pixels", f"{mask_pixels:,}")
                with col_stats2:
                    st.metric("Mask Coverage", f"{mask_percentage:.1f}%")
                with col_stats3:
                    st.metric("Max Confidence", f"{np.max(probability_mask):.3f}")
                with col_stats4:
                    st.metric("Mean Confidence", f"{np.mean(probability_mask):.3f}")
                
                # Download section
                st.subheader("💾 Download Results")
                
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                
                with col_dl1:
                    # Binary mask
                    mask_pil = Image.fromarray(binary_mask)
                    buf_mask = io.BytesIO()
                    mask_pil.save(buf_mask, format='PNG')
                    st.download_button(
                        label="Download Binary Mask",
                        data=buf_mask.getvalue(),
                        file_name="unet_mask.png",
                        mime="image/png"
                    )
                
                with col_dl2:
                    # Probability map
                    prob_display = (probability_mask * 255).astype(np.uint8)
                    prob_pil = Image.fromarray(prob_display)
                    buf_prob = io.BytesIO()
                    prob_pil.save(buf_prob, format='PNG')
                    st.download_button(
                        label="Download Probability Map",
                        data=buf_prob.getvalue(),
                        file_name="probability_map.png",
                        mime="image/png"
                    )
                
                with col_dl3:
                    # Overlay
                    overlay_pil = Image.fromarray(overlay)
                    buf_overlay = io.BytesIO()
                    overlay_pil.save(buf_overlay, format='PNG')
                    st.download_button(
                        label="Download Overlay",
                        data=buf_overlay.getvalue(),
                        file_name="mask_overlay.png",
                        mime="image/png"
                    )
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.code(f"Error details: {str(e)}")

else:
    st.warning("""
    ⚠️ **U-Net Model not loaded!**
    
    Please ensure your model file is in the repository with one of these names:
    - `unet_model.pth`, `unet_model.pt`
    - `model.pth`, `model.pt`
    - `unet.pth`, `segmentation_model.pth`
    
    **If your U-Net architecture is different**, you may need to:
    1. Update the model class definition above
    2. Adjust the input/output channels
    3. Match your exact architecture
    """)

# Model architecture info
with st.expander("🔧 U-Net Architecture Information"):
    st.markdown("""
    **Current U-Net Configuration:**
    - Input channels: 3 (RGB)
    - Output channels: 1 (Binary segmentation)
    - Base features: 64
    - Activation: Sigmoid (for binary segmentation)
    
    **If your model is different:**
    - Change `in_channels` and `out_channels` in the UNet class
    - Adjust the architecture to match your trained model
    - Update preprocessing/postprocessing as needed
    """)

# Instructions for custom U-Net
with st.expander("📚 How to adapt for your U-Net"):
    st.markdown("""
    ```python
    # If your U-Net has different parameters:
    class YourUNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=2, features=32):
            # Your architecture here
            pass
    
    # Update the model loading:
    model = YourUNet(in_channels=3, out_channels=2)
    ```
    
    **For multi-class segmentation:**
    - Change `out_channels` to number of classes
    - Use softmax instead of sigmoid
    - Update postprocessing for multi-class
    """)
