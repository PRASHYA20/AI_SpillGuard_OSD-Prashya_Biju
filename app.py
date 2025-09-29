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

# Define the EXACT architecture that matches your state dict
class ExactOilSpillModel(nn.Module):
    def __init__(self, num_classes=1):
        super(ExactOilSpillModel, self).__init__()
        
        # Encoder - ResNet50 (matches your state dict exactly)
        resnet = models.resnet50(pretrained=False)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Decoder blocks (matching your state dict structure)
        self.decoder = nn.ModuleList([
            self._make_decoder_block(2048, 1024),  # block 0
            self._make_decoder_block(1024, 512),   # block 1
            self._make_decoder_block(512, 256),    # block 2
            self._make_decoder_block(256, 128),    # block 3
            self._make_decoder_block(128, 64),     # block 4
        ])
        
        # Segmentation head (matches 'segmentation_head' in state dict)
        self.segmentation_head = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        
        # Decoder
        for block in self.decoder:
            x = block(x)
            # Add upsampling if needed (adjust based on your architecture)
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Segmentation head
        x = self.segmentation_head(x)
        return torch.sigmoid(x)

@st.cache_resource
def load_exact_model():
    """Load the model with exact architecture matching"""
    try:
        # Initialize model with exact architecture
        model = ExactOilSpillModel(num_classes=1)
        
        # Load state dict
        checkpoint = torch.load('oil_spill_model_deploy.pth', map_location='cpu')
        
        # Load state dict with strict=False to handle minor mismatches
        model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        st.sidebar.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None

# Alternative: Try loading without defining architecture
@st.cache_resource  
def load_model_directly():
    """Try to load the model directly"""
    try:
        # This will work if the model was saved with torch.save(model, ...)
        model = torch.load('oil_spill_model_deploy.pth', map_location='cpu')
        
        if isinstance(model, torch.nn.Module):
            model.eval()
            st.sidebar.success("✅ Full model loaded directly!")
            return model
        else:
            st.sidebar.warning("⚠️ Loaded state dict, need architecture")
            return None
            
    except Exception as e:
        st.sidebar.error(f"❌ Direct loading failed: {str(e)}")
        return None

# Try both loading methods
model = load_model_directly()
if model is None:
    model = load_exact_model()

# Model info
st.sidebar.header("Model Information")
if model is not None:
    st.sidebar.success("✅ Oil Spill Model Loaded")
    total_params = sum(p.numel() for p in model.parameters())
    st.sidebar.write(f"**Parameters:** {total_params:,}")
else:
    st.sidebar.error("❌ Model not loaded")

# Image preprocessing
def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for model input"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    original_size = image.size
    image = image.resize(target_size)
    
    # Convert to numpy and normalize with ImageNet stats
    img_array = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, original_size

def postprocess_mask(mask, original_size, threshold=0.5):
    """Postprocess model output mask"""
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().cpu().numpy()
    
    # Apply threshold
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    return mask_resized, mask

def create_oil_spill_visualization(original_image, mask, alpha=0.6):
    """Create oil spill visualization"""
    original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    
    # Create colored mask (red for oil spills)
    colored_mask = np.zeros_like(original_cv)
    colored_mask[mask > 0] = [0, 0, 255]  # Red color
    
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
    step=0.1
)

# Main app
if model is not None:
    st.header("📡 Upload Satellite Imagery")
    
    uploaded_file = st.file_uploader(
        "Choose satellite image", 
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp']
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
                
                with col3:
                    st.subheader("🛢️ Oil Spill Overlay")
                    overlay = create_oil_spill_visualization(image, binary_mask)
                    st.image(overlay, use_column_width=True)
                
                # Statistics
                st.subheader("📊 Oil Spill Analysis")
                
                total_pixels = binary_mask.size
                oil_pixels = np.sum(binary_mask > 0)
                oil_percentage = (oil_pixels / total_pixels) * 100
                
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                
                with col_stats1:
                    st.metric("Oil Spill Area", f"{oil_pixels:,} px")
                with col_stats2:
                    st.metric("Coverage", f"{oil_percentage:.2f}%")
                with col_stats3:
                    st.metric("Confidence", f"{np.max(probability_mask):.3f}")
                
                # Alert
                if oil_pixels > 0:
                    st.warning(f"🚨 Oil spill detected! {oil_percentage:.2f}% coverage")
                else:
                    st.success("✅ No oil spills detected")
                
                # Download
                st.subheader("💾 Download Results")
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    mask_pil = Image.fromarray(binary_mask)
                    buf_mask = io.BytesIO()
                    mask_pil.save(buf_mask, format='PNG')
                    st.download_button(
                        label="Download Mask",
                        data=buf_mask.getvalue(),
                        file_name="oil_spill_mask.png",
                        mime="image/png"
                    )
                
                with col_dl2:
                    overlay_pil = Image.fromarray(overlay)
                    buf_overlay = io.BytesIO()
                    overlay_pil.save(buf_overlay, format='PNG')
                    st.download_button(
                        label="Download Overlay",
                        data=buf_overlay.getvalue(),
                        file_name="oil_spill_overlay.png",
                        mime="image/png"
                    )
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")

else:
    st.error("""
    ❌ Model failed to load!
    
    **Please try this quick fix:**
    
    If you have the original training code, find how the model was defined and replace the `ExactOilSpillModel` class with your exact architecture.
    
    **Common solutions:**
    1. Use the exact model class from your training script
    2. Or share your model architecture code so I can match it exactly
    """)

# Debug information
with st.expander("🔧 Debug Info"):
    st.write("**Model File:** `oil_spill_model_deploy.pth`")
    if model is not None:
        st.write("**Model Type:**", type(model))
        st.write("**Model Device:**", next(model.parameters()).device)
