import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import io
import collections

# Set page config
st.set_page_config(page_title="Oil Spill Detection", layout="wide")

st.title("🛢️ Oil Spill Detection")
st.write("Upload satellite imagery to detect oil spills")

# Define a proper U-Net like architecture that matches common ResNet-based segmentation models
class OilSpillSegmentationModel(nn.Module):
    def __init__(self, num_classes=1):
        super(OilSpillSegmentationModel, self).__init__()
        
        # Encoder - ResNet50 backbone
        self.encoder = models.resnet50(pretrained=False)
        
        # Remove the final fully connected layer
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        
        # Decoder - simple upsampling path
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Final segmentation head
        self.segmentation_head = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        
        # Decoder
        x = self.decoder(x)
        
        # Segmentation head
        x = self.segmentation_head(x)
        
        # Upsample to input size and apply sigmoid
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return torch.sigmoid(x)

@st.cache_resource
def load_model_properly():
    """Load model properly handling state dict vs full model"""
    try:
        # First, try to load the file
        checkpoint = torch.load('oil_spill_model_deploy.pth', map_location='cpu')
        
        st.sidebar.info(f"📦 Loaded file type: {type(checkpoint)}")
        
        # Case 1: It's a state dictionary (OrderedDict)
        if isinstance(checkpoint, collections.OrderedDict):
            st.sidebar.info("🔄 Loading state dictionary into model architecture...")
            
            # Initialize model with proper architecture
            model = OilSpillSegmentationModel(num_classes=1)
            
            # Try to load state dict
            try:
                model.load_state_dict(checkpoint)
                st.sidebar.success("✅ State dict loaded successfully!")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Standard loading failed: {e}")
                st.sidebar.info("🔄 Trying strict=False loading...")
                
                # Try with strict=False to handle minor mismatches
                model.load_state_dict(checkpoint, strict=False)
                st.sidebar.success("✅ State dict loaded with strict=False!")
            
            model.eval()
            return model
            
        # Case 2: It's already a model instance
        elif isinstance(checkpoint, torch.nn.Module):
            st.sidebar.success("✅ Full model loaded directly!")
            checkpoint.eval()
            return checkpoint
            
        else:
            st.sidebar.error(f"❌ Unknown checkpoint type: {type(checkpoint)}")
            return None
            
    except Exception as e:
        st.sidebar.error(f"❌ Model loading error: {str(e)}")
        return None

# Load the model
model = load_model_properly()

# Simple preprocessing
def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for model input"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    original_size = image.size
    image_resized = image.resize(target_size)
    
    # Convert to numpy and normalize with ImageNet stats
    img_array = np.array(image_resized) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, original_size

def clean_mask(mask, min_size=500):
    """Remove small noisy detections"""
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_mask[labels == i] = 255
    
    return cleaned_mask

# Settings
st.sidebar.header("⚙️ Detection Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    0.1, 0.9, 0.5, 0.05,
    help="Higher values = fewer false positives, but might miss small spills"
)

enable_cleaning = st.sidebar.checkbox("Clean Mask (remove small detections)", value=True)
min_object_size = st.sidebar.slider("Minimum Object Size (pixels)", 100, 2000, 500, 100)

# Alternative: Try different model architectures if the first one fails
if model is None:
    st.sidebar.warning("🔄 Trying alternative model architecture...")
    
    # Try a simpler architecture
    class SimpleSegmentationModel(nn.Module):
        def __init__(self):
            super(SimpleSegmentationModel, self).__init__()
            self.encoder = models.resnet34(pretrained=False)
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
            self.decoder = nn.Conv2d(512, 1, kernel_size=1)
            
        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            x = nn.functional.interpolate(x, scale_factor=32, mode='bilinear', align_corners=False)
            return torch.sigmoid(x)
    
    try:
        checkpoint = torch.load('oil_spill_model_deploy.pth', map_location='cpu')
        model = SimpleSegmentationModel()
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        st.sidebar.success("✅ Alternative architecture loaded!")
    except:
        st.sidebar.error("❌ All loading attempts failed")

# Main application
if model is not None:
    st.header("📡 Upload Satellite Imagery")
    
    uploaded_file = st.file_uploader(
        "Choose satellite image", 
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp']
    )
    
    if uploaded_file is not None:
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
                
                # Get model device
                device = next(model.parameters()).device
                input_tensor = input_tensor.to(device)
                
                # Prediction
                with torch.no_grad():
                    output = model(input_tensor)
                
                # Handle different output formats
                if isinstance(output, (list, tuple)):
                    output = output[0]  # Take first output if multiple
                
                # Convert to probability mask
                probability_mask = output.squeeze().cpu().numpy()
                
                # Create binary mask
                binary_mask = (probability_mask > confidence_threshold).astype(np.uint8) * 255
                binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
                
                # Clean mask if enabled
                if enable_cleaning:
                    binary_mask_resized = clean_mask(binary_mask_resized, min_object_size)
                
                # Create overlay
                original_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                colored_mask = np.zeros_like(original_cv)
                colored_mask[binary_mask_resized > 0] = [0, 0, 255]  # Red for oil spills
                blended = cv2.addWeighted(original_cv, 0.7, colored_mask, 0.3, 0)
                blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
                
                # Display results
                with col2:
                    st.subheader("🎭 Detection Mask")
                    st.image(binary_mask_resized, use_column_width=True, clamp=True)
                    st.write(f"Threshold: {confidence_threshold}")
                
                with col3:
                    st.subheader("🛢️ Oil Spill Overlay")
                    st.image(blended_rgb, use_column_width=True)
                
                # Statistics
                st.subheader("📊 Detection Results")
                
                total_pixels = binary_mask_resized.size
                oil_pixels = np.sum(binary_mask_resized > 0)
                oil_percentage = (oil_pixels / total_pixels) * 100
                
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                
                with col_stats1:
                    st.metric("Oil Spill Area", f"{oil_pixels:,} px")
                with col_stats2:
                    st.metric("Coverage", f"{oil_percentage:.2f}%")
                with col_stats3:
                    confidence_max = np.max(probability_mask)
                    st.metric("Max Confidence", f"{confidence_max:.3f}")
                
                # Alert system
                st.subheader("🚨 Detection Alert")
                if oil_pixels > 0:
                    if oil_percentage > 5:
                        st.error(f"🚨 LARGE OIL SPILL DETECTED! {oil_percentage:.2f}% coverage")
                    elif oil_percentage > 1:
                        st.warning(f"⚠️ Medium oil spill detected: {oil_percentage:.2f}% coverage")
                    else:
                        st.info(f"ℹ️ Small oil spill detected: {oil_percentage:.2f}% coverage")
                else:
                    st.success("✅ No oil spills detected")
                
                # Model confidence analysis
                st.subheader("🔍 Model Confidence Analysis")
                conf_col1, conf_col2 = st.columns(2)
                
                with conf_col1:
                    st.write("**Confidence Statistics:**")
                    st.write(f"- Min: {np.min(probability_mask):.4f}")
                    st.write(f"- Max: {np.max(probability_mask):.4f}")
                    st.write(f"- Mean: {np.mean(probability_mask):.4f}")
                    st.write(f"- Std: {np.std(probability_mask):.4f}")
                
                with conf_col2:
                    st.write("**Interpretation:**")
                    if np.max(probability_mask) < 0.1:
                        st.error("❌ Model is very uncertain")
                    elif np.mean(probability_mask) > 0.8:
                        st.warning("⚠️ Model is overconfident")
                    else:
                        st.success("✅ Model confidence looks reasonable")
                
                # Quick threshold comparison
                st.subheader("🔍 Quick Threshold Test")
                st.write("Try different thresholds to find the best one:")
                
                test_thresholds = [0.3, 0.5, 0.7]
                test_cols = st.columns(3)
                
                for col, test_thresh in zip(test_cols, test_thresholds):
                    with col:
                        test_mask = (probability_mask > test_thresh).astype(np.uint8) * 255
                        test_mask_resized = cv2.resize(test_mask, original_size, interpolation=cv2.INTER_NEAREST)
                        
                        if enable_cleaning:
                            test_mask_resized = clean_mask(test_mask_resized, min_object_size)
                        
                        test_pixels = np.sum(test_mask_resized > 0)
                        test_coverage = (test_pixels / total_pixels) * 100
                        
                        st.image(test_mask_resized, use_column_width=True, caption=f"Thresh: {test_thresh}")
                        st.write(f"Coverage: {test_coverage:.2f}%")
                
                # Download section
                st.subheader("💾 Download Results")
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    mask_pil = Image.fromarray(binary_mask_resized)
                    buf_mask = io.BytesIO()
                    mask_pil.save(buf_mask, format='PNG')
                    st.download_button(
                        label="Download Detection Mask",
                        data=buf_mask.getvalue(),
                        file_name="oil_spill_mask.png",
                        mime="image/png"
                    )
                
                with col_dl2:
                    overlay_pil = Image.fromarray(blended_rgb)
                    buf_overlay = io.BytesIO()
                    overlay_pil.save(buf_overlay, format='PNG')
                    st.download_button(
                        label="Download Overlay Image",
                        data=buf_overlay.getvalue(),
                        file_name="oil_spill_overlay.png", 
                        mime="image/png"
                    )
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("💡 Try adjusting the threshold or disabling mask cleaning")

else:
    st.error("""
    ❌ Model failed to load!
    
    **This usually means:**
    1. The model file format is not compatible
    2. We need the exact model architecture used during training
    3. The model file might be corrupted
    
    **Solution:** Please share the original model architecture code from your training script.
    """)

# Tips for better detection
with st.expander("💡 Tips for Better Detection"):
    st.markdown("""
    **If detections are wrong:**
    
    - **Too many false positives?** → Increase threshold (0.7-0.9)
    - **Missing real spills?** → Decrease threshold (0.3-0.5)
    - **Noisy output?** → Enable mask cleaning and increase min object size
    - **Model uncertain?** → Try different preprocessing or check input image quality
    
    **Recommended settings to start:**
    - Threshold: 0.5
    - Mask cleaning: Enabled
    - Min object size: 500 pixels
    """)
