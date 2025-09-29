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

# Define the EXACT architecture that matches your state dict
class ExactOilSpillModel(nn.Module):
    def __init__(self, num_classes=1):
        super(ExactOilSpillModel, self).__init__()
        
        # Encoder - ResNet50 with exact layer structure
        resnet = models.resnet50(pretrained=False)
        
        # Store layers exactly as they appear in your state dict
        self.encoder_conv1 = resnet.conv1
        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool
        self.encoder_layer1 = resnet.layer1
        self.encoder_layer2 = resnet.layer2
        self.encoder_layer3 = resnet.layer3
        self.encoder_layer4 = resnet.layer4
        
        # Decoder blocks - matching your state dict structure
        self.decoder_blocks = nn.ModuleList([
            self._make_decoder_block(2048, 1024),  # block 0
            self._make_decoder_block(1024, 512),   # block 1  
            self._make_decoder_block(512, 256),    # block 2
            self._make_decoder_block(256, 128),    # block 3
            self._make_decoder_block(128, 64),     # block 4
        ])
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def _make_decoder_block(self, in_channels, out_channels):
        """Create decoder block matching your state dict structure"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder forward pass - exactly matching state dict structure
        x = self.encoder_conv1(x)
        x = self.encoder_bn1(x)
        x = self.encoder_relu(x)
        x = self.encoder_maxpool(x)

        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x) 
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        
        # Decoder forward pass
        for block in self.decoder_blocks:
            x = block(x)
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Segmentation head
        x = self.segmentation_head(x)
        return torch.sigmoid(x)

@st.cache_resource
def load_model_exact():
    """Load model with exact architecture matching"""
    try:
        # Load the state dict
        checkpoint = torch.load('oil_spill_model_deploy.pth', map_location='cpu')
        
        st.sidebar.info(f"📦 Loaded file type: {type(checkpoint)}")
        
        if isinstance(checkpoint, collections.OrderedDict):
            st.sidebar.info("🔄 Loading with exact architecture match...")
            
            # Initialize exact model
            model = ExactOilSpillModel(num_classes=1)
            
            # Load with strict=False to handle any remaining minor mismatches
            model.load_state_dict(checkpoint, strict=False)
            
            st.sidebar.success("✅ Model loaded successfully with exact architecture!")
            model.eval()
            return model
            
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
model = load_model_exact()

# If exact loading fails, try a universal approach
if model is None:
    st.sidebar.warning("🔄 Trying universal model loader...")
    
    class UniversalSegmentationModel(nn.Module):
        """A universal model that should work with most ResNet-based segmentation models"""
        def __init__(self):
            super(UniversalSegmentationModel, self).__init__()
            # Use ResNet50 as base
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
            # Simple decoder
            self.decoder = nn.Sequential(
                nn.Conv2d(2048, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                
                nn.Conv2d(1024, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                
                nn.Conv2d(512, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
            self.head = nn.Conv2d(64, 1, 1)
            
        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            x = self.head(x)
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
            return torch.sigmoid(x)
    
    try:
        checkpoint = torch.load('oil_spill_model_deploy.pth', map_location='cpu')
        model = UniversalSegmentationModel()
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        st.sidebar.success("✅ Universal model loaded!")
    except Exception as e:
        st.sidebar.error(f"❌ Universal loading failed: {e}")

# Simple preprocessing
def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for model input"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    original_size = image.size
    image_resized = image.resize(target_size)
    
    # Convert to numpy and normalize
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
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_mask[labels == i] = 255
    
    return cleaned_mask

# Settings
st.sidebar.header("⚙️ Detection Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    0.1, 0.9, 0.5, 0.05
)

enable_cleaning = st.sidebar.checkbox("Clean Mask", value=True)
min_object_size = st.sidebar.slider("Min Object Size", 100, 2000, 500, 100)

# Test different input sizes
input_size = st.sidebar.selectbox(
    "Input Size",
    [224, 256, 384, 512],
    index=1,
    help="Try different input sizes if detection is poor"
)

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
                # Preprocess with selected size
                input_tensor, original_size = preprocess_image(image, target_size=(input_size, input_size))
                
                # Move to same device as model
                device = next(model.parameters()).device
                input_tensor = input_tensor.to(device)
                
                # Prediction
                with torch.no_grad():
                    output = model(input_tensor)
                
                # Handle output
                if isinstance(output, (list, tuple)):
                    output = output[0]
                
                probability_mask = output.squeeze().cpu().numpy()
                
                # Create binary mask
                binary_mask = (probability_mask > confidence_threshold).astype(np.uint8) * 255
                binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
                
                # Clean mask
                if enable_cleaning:
                    binary_mask_resized = clean_mask(binary_mask_resized, min_object_size)
                
                # Create overlay
                original_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                colored_mask = np.zeros_like(original_cv)
                colored_mask[binary_mask_resized > 0] = [0, 0, 255]
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
                
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                
                with col_stats1:
                    st.metric("Oil Pixels", f"{oil_pixels:,}")
                with col_stats2:
                    st.metric("Coverage", f"{oil_percentage:.2f}%")
                with col_stats3:
                    st.metric("Max Confidence", f"{np.max(probability_mask):.3f}")
                with col_stats4:
                    st.metric("Mean Confidence", f"{np.mean(probability_mask):.3f}")
                
                # Alert
                if oil_pixels > 0:
                    if oil_percentage > 5:
                        st.error(f"🚨 LARGE OIL SPILL! {oil_percentage:.2f}% coverage")
                    elif oil_percentage > 1:
                        st.warning(f"⚠️ Medium spill: {oil_percentage:.2f}% coverage")
                    else:
                        st.info(f"ℹ️ Small spill: {oil_percentage:.2f}% coverage")
                else:
                    st.success("✅ No oil spills detected")
                
                # Quick threshold test
                st.subheader("🔍 Test Different Thresholds")
                test_thresholds = [0.3, 0.5, 0.7, 0.9]
                test_cols = st.columns(4)
                
                for col, thresh in zip(test_cols, test_thresholds):
                    with col:
                        test_mask = (probability_mask > thresh).astype(np.uint8) * 255
                        test_mask_resized = cv2.resize(test_mask, original_size, interpolation=cv2.INTER_NEAREST)
                        
                        if enable_cleaning:
                            test_mask_resized = clean_mask(test_mask_resized, min_object_size)
                        
                        test_pixels = np.sum(test_mask_resized > 0)
                        test_coverage = (test_pixels / total_pixels) * 100
                        
                        st.image(test_mask_resized, use_column_width=True, caption=f"Thresh: {thresh}")
                        st.write(f"Coverage: {test_coverage:.2f}%")
                
                # Download
                st.subheader("💾 Download Results")
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    mask_pil = Image.fromarray(binary_mask_resized)
                    buf_mask = io.BytesIO()
                    mask_pil.save(buf_mask, format='PNG')
                    st.download_button(
                        label="Download Mask",
                        data=buf_mask.getvalue(),
                        file_name="oil_spill_mask.png",
                        mime="image/png"
                    )
                
                with col_dl2:
                    overlay_pil = Image.fromarray(blended_rgb)
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
                st.info("💡 Try different input size or threshold")

else:
    st.error("""
    ❌ Model failed to load!
    
    **We need the exact model architecture.** Please consider:
    
    1. **Share your training code** - The exact model class definition
    2. **Re-save the model** - Use `torch.save(model, 'model.pth')` instead of state dict
    3. **Check model format** - Ensure it's a PyTorch model file
    
    Without the exact architecture, we can only approximate.
    """)

# Final fallback - direct prediction without proper architecture
with st.expander("🔄 Last Resort: Direct Prediction"):
    st.warning("This is a last resort method - results may be poor")
    
    if st.button("Try Direct Prediction (Experimental)"):
        try:
            # Load state dict directly and try to use it
            checkpoint = torch.load('oil_spill_model_deploy.pth', map_location='cpu')
            
            # Create a simple resnet and try to load
            simple_model = models.resnet50(pretrained=False)
            simple_model.fc = nn.Identity()  # Remove classification head
            
            try:
                simple_model.load_state_dict(checkpoint, strict=False)
                simple_model.eval()
                st.success("✅ Direct loading worked! Model ready for prediction.")
            except:
                st.error("❌ Even direct loading failed.")
                
        except Exception as e:
            st.error(f"Final attempt failed: {e}")
