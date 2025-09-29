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

# Define model architecture that matches common segmentation models
class OilSpillModel(nn.Module):
    def __init__(self, num_classes=1):
        super(OilSpillModel, self).__init__()
        
        # Encoder - ResNet50
        resnet = models.resnet50(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Final output
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return torch.sigmoid(x)

@st.cache_resource
def load_model():
    """Load model with proper error handling"""
    try:
        checkpoint = torch.load('oil_spill_model_deploy.pth', map_location='cpu')
        
        if isinstance(checkpoint, collections.OrderedDict):
            # It's a state dict - load into model architecture
            model = OilSpillModel(num_classes=1)
            
            # Try to load state dict
            try:
                model.load_state_dict(checkpoint)
                st.sidebar.success("✅ Model loaded successfully!")
            except:
                # If strict loading fails, try with strict=False
                model.load_state_dict(checkpoint, strict=False)
                st.sidebar.success("✅ Model loaded with strict=False")
            
            model.eval()
            return model
            
        elif isinstance(checkpoint, torch.nn.Module):
            # It's already a model
            checkpoint.eval()
            st.sidebar.success("✅ Full model loaded directly!")
            return checkpoint
            
    except Exception as e:
        st.sidebar.error(f"❌ Loading failed: {str(e)}")
        return None

# Load model
model = load_model()

# Enhanced preprocessing with multiple options
def preprocess_image_advanced(image, target_size=(512, 512), method='standard'):
    """Advanced preprocessing with different strategies"""
    original_size = image.size
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(image_resized)
    
    # Different preprocessing strategies
    if method == 'no_normalize':
        # Simple 0-1 normalization
        img_array = img_array.astype(np.float32) / 255.0
        
    elif method == 'simple_normalize':
        # Simple normalization with mean subtraction
        img_array = img_array.astype(np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5
        
    elif method == 'opencv_style':
        # BGR and ImageNet stats (if model was trained with OpenCV)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_array = img_array.astype(np.float32) / 255.0
        mean = np.array([0.406, 0.456, 0.485])  # BGR order
        std = np.array([0.225, 0.224, 0.229])
        img_array = (img_array - mean) / std
        
    else:  # standard
        # Standard ImageNet normalization (RGB)
        img_array = img_array.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, original_size

def analyze_model_output(output):
    """Comprehensive analysis of model output"""
    if isinstance(output, torch.Tensor):
        prob_map = output.squeeze().cpu().numpy()
    else:
        prob_map = output
    
    analysis = {
        'min': float(np.min(prob_map)),
        'max': float(np.max(prob_map)),
        'mean': float(np.mean(prob_map)),
        'std': float(np.std(prob_map)),
        'median': float(np.median(prob_map)),
        'q95': float(np.percentile(prob_map, 95)),
        'q99': float(np.percentile(prob_map, 99)),
        'above_0.1': float(np.sum(prob_map > 0.1) / prob_map.size * 100),
        'above_0.5': float(np.sum(prob_map > 0.5) / prob_map.size * 100),
        'above_0.8': float(np.sum(prob_map > 0.8) / prob_map.size * 100),
    }
    return analysis, prob_map

def calculate_optimal_threshold(prob_map, method='otsu'):
    """Calculate optimal threshold using different methods"""
    prob_255 = (prob_map * 255).astype(np.uint8)
    
    if method == 'otsu' and len(np.unique(prob_255)) > 1:
        try:
            threshold, _ = cv2.threshold(prob_255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return threshold / 255.0
        except:
            pass
    
    elif method == 'mean_std':
        mean = np.mean(prob_map)
        std = np.std(prob_map)
        return min(0.9, max(0.1, mean + std))
    
    elif method == 'percentile':
        return np.percentile(prob_map, 95)
    
    # Fallback to mean-based threshold
    return min(0.8, max(0.3, np.mean(prob_map) * 1.5))

def enhance_mask(mask, prob_map, enhancement='standard'):
    """Enhance the binary mask"""
    if enhancement == 'morphological':
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    elif enhancement == 'confidence_weighted':
        # Create smoother edges based on confidence
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        enhanced_mask = np.zeros_like(mask)
        for contour in contours:
            # Get bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            # Calculate average confidence in this region
            region_conf = np.mean(prob_map[y:y+h, x:x+w])
            if region_conf > 0.3:  # Only keep high-confidence regions
                cv2.fillPoly(enhanced_mask, [contour], 255)
        mask = enhanced_mask
    
    return mask

def clean_mask(mask, min_size=1000, max_size=None):
    """Clean mask by removing small objects and optionally large ones"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Keep only objects within size range
        if area >= min_size:
            if max_size is None or area <= max_size:
                cleaned_mask[labels == i] = 255
    
    return cleaned_mask

# Settings
st.sidebar.header("⚙️ Advanced Settings")

# Preprocessing options
preprocessing_method = st.sidebar.selectbox(
    "Preprocessing Method",
    ['standard', 'no_normalize', 'simple_normalize', 'opencv_style'],
    help="Try different preprocessing if detection is poor"
)

input_size = st.sidebar.selectbox(
    "Input Size",
    [256, 384, 512, 768],
    index=2,
    help="Larger sizes may detect smaller spills better"
)

# Threshold options
threshold_method = st.sidebar.radio(
    "Threshold Method",
    ['auto_otsu', 'auto_percentile', 'manual']
)

if threshold_method == 'manual':
    confidence_threshold = st.sidebar.slider("Manual Threshold", 0.01, 0.99, 0.5, 0.01)
else:
    confidence_threshold = None

# Post-processing
st.sidebar.subheader("Post-processing")
enable_cleaning = st.sidebar.checkbox("Clean Mask", value=True)
min_object_size = st.sidebar.slider("Min Object Size", 500, 5000, 1000, 100)
enable_enhancement = st.sidebar.checkbox("Enhance Mask", value=True)

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
        
        # Display original image
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🛰️ Original Image")
            st.image(image, use_container_width=True)
            st.write(f"Size: {original_size}")
        
        # Process and predict
        with st.spinner("🔍 Analyzing for oil spills..."):
            try:
                # Preprocess
                input_tensor, original_size = preprocess_image_advanced(
                    image, 
                    target_size=(input_size, input_size),
                    method=preprocessing_method
                )
                
                # Move to model device
                device = next(model.parameters()).device
                input_tensor = input_tensor.to(device)
                
                # Prediction
                with torch.no_grad():
                    output = model(input_tensor)
                
                # Handle output
                if isinstance(output, (list, tuple)):
                    output = output[0]
                
                # Analyze output
                analysis, prob_map = analyze_model_output(output)
                
                # Calculate threshold
                if threshold_method == 'auto_otsu':
                    confidence_threshold = calculate_optimal_threshold(prob_map, 'otsu')
                elif threshold_method == 'auto_percentile':
                    confidence_threshold = calculate_optimal_threshold(prob_map, 'percentile')
                
                st.info(f"**Using threshold: {confidence_threshold:.3f}**")
                
                # Create initial mask
                binary_mask = (prob_map > confidence_threshold).astype(np.uint8) * 255
                binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
                
                # Apply enhancements
                if enable_enhancement:
                    binary_mask_resized = enhance_mask(binary_mask_resized, prob_map)
                
                # Clean mask
                if enable_cleaning:
                    binary_mask_resized = clean_mask(binary_mask_resized, min_object_size)
                
                # Create overlay
                original_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                colored_mask = np.zeros_like(original_cv)
                colored_mask[binary_mask_resized > 0] = [0, 0, 255]  # Red for oil
                blended = cv2.addWeighted(original_cv, 0.7, colored_mask, 0.3, 0)
                blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
                
                # Display results
                with col2:
                    st.subheader("🎭 Detection Mask")
                    st.image(binary_mask_resized, use_container_width=True)
                    st.write(f"Threshold: {confidence_threshold:.3f}")
                
                with col3:
                    st.subheader("🛢️ Oil Spill Overlay")
                    st.image(blended_rgb, use_container_width=True)
                
                # Detailed analysis
                st.subheader("📊 Detailed Analysis")
                
                # Statistics
                total_pixels = binary_mask_resized.size
                oil_pixels = np.sum(binary_mask_resized > 0)
                oil_percentage = (oil_pixels / total_pixels) * 100
                
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                
                with col_stats1:
                    st.metric("Oil Pixels", f"{oil_pixels:,}")
                with col_stats2:
                    st.metric("Coverage", f"{oil_percentage:.4f}%")
                with col_stats3:
                    st.metric("Max Confidence", f"{analysis['max']:.3f}")
                with col_stats4:
                    st.metric("Mean Confidence", f"{analysis['mean']:.3f}")
                
                # Model output analysis
                st.subheader("🔍 Model Output Analysis")
                anal_col1, anal_col2 = st.columns(2)
                
                with anal_col1:
                    st.write("**Confidence Distribution:**")
                    st.write(f"- Min: {analysis['min']:.4f}")
                    st.write(f"- Max: {analysis['max']:.4f}")
                    st.write(f"- Mean: {analysis['mean']:.4f}")
                    st.write(f"- Std: {analysis['std']:.4f}")
                    st.write(f"- Median: {analysis['median']:.4f}")
                    st.write(f"- 95th %ile: {analysis['q95']:.4f}")
                
                with anal_col2:
                    st.write("**Pixel Statistics:**")
                    st.write(f"- > 0.1: {analysis['above_0.1']:.2f}%")
                    st.write(f"- > 0.5: {analysis['above_0.5']:.2f}%")
                    st.write(f"- > 0.8: {analysis['above_0.8']:.2f}%")
                    
                    # Interpretation
                    st.write("**Interpretation:**")
                    if analysis['max'] < 0.1:
                        st.error("❌ Model very uncertain - try different preprocessing")
                    elif analysis['mean'] > 0.8:
                        st.warning("⚠️ Model overconfident - try higher threshold")
                    elif oil_pixels == 0:
                        st.info("ℹ️ No spills detected - try lower threshold")
                    else:
                        st.success("✅ Model producing reasonable outputs")
                
                # Alert system
                st.subheader("🚨 Detection Alert")
                if oil_pixels > 0:
                    if oil_percentage > 1:
                        st.error(f"🚨 OIL SPILL DETECTED! {oil_percentage:.4f}% coverage")
                    else:
                        st.warning(f"⚠️ Potential oil sheen: {oil_percentage:.4f}% coverage")
                else:
                    st.success("✅ No oil spills detected")
                
                # Quick threshold comparison
                st.subheader("🔍 Compare Thresholds")
                test_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
                test_cols = st.columns(5)
                
                for col, thresh in zip(test_cols, test_thresholds):
                    with col:
                        test_mask = (prob_map > thresh).astype(np.uint8) * 255
                        test_mask_resized = cv2.resize(test_mask, original_size, interpolation=cv2.INTER_NEAREST)
                        
                        if enable_cleaning:
                            test_mask_resized = clean_mask(test_mask_resized, min_object_size)
                        
                        test_pixels = np.sum(test_mask_resized > 0)
                        test_coverage = (test_pixels / total_pixels) * 100
                        
                        st.image(test_mask_resized, use_container_width=True, caption=f"Thresh: {thresh}")
                        st.write(f"Coverage: {test_coverage:.4f}%")
                
                # Download section
                st.subheader("💾 Download Results")
                
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                
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
                
                with col_dl3:
                    # Analysis report
                    report = f"""OIL SPILL DETECTION REPORT
Image Size: {original_size}
Total Pixels: {total_pixels:,}
Oil Spill Pixels: {oil_pixels:,}
Area Coverage: {oil_percentage:.4f}%
Threshold: {confidence_threshold:.3f}

MODEL ANALYSIS:
Min Confidence: {analysis['min']:.4f}
Max Confidence: {analysis['max']:.4f}
Mean Confidence: {analysis['mean']:.4f}
Std Confidence: {analysis['std']:.4f}
Pixels > 0.1: {analysis['above_0.1']:.2f}%
Pixels > 0.5: {analysis['above_0.5']:.2f}%
Pixels > 0.8: {analysis['above_0.8']:.2f}%

SETTINGS:
Preprocessing: {preprocessing_method}
Input Size: {input_size}
Threshold Method: {threshold_method}
Mask Cleaning: {enable_cleaning}
Min Object Size: {min_object_size}
"""
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name="oil_spill_analysis.txt",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")

else:
    st.error("Model failed to load. Please check the model file.")

# Troubleshooting guide
with st.expander("🔧 Troubleshooting Guide"):
    st.markdown("""
    **If detection is poor:**
    
    1. **Try different preprocessing methods:**
       - `standard`: ImageNet normalization (most common)
       - `no_normalize`: Simple 0-1 scaling
       - `opencv_style`: BGR format with ImageNet stats
    
    2. **Adjust input size:**
       - Larger sizes (768px) for small spills
       - Smaller sizes (256px) for computational efficiency
    
    3. **Use auto-threshold methods:**
       - `auto_otsu`: Automatic threshold using Otsu's method
       - `auto_percentile`: Uses 95th percentile of confidence
    
    4. **Post-processing:**
       - Enable mask cleaning to remove noise
       - Adjust min object size based on expected spill size
       - Try mask enhancement for smoother results
    
    **Start with these settings:**
    - Preprocessing: `standard`
    - Input Size: `512`
    - Threshold: `auto_otsu`
    - Mask Cleaning: `Enabled`
    - Min Object Size: `1000`
    """)
