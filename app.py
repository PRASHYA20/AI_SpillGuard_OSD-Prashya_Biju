import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import io
import collections
import os

# Set page config
st.set_page_config(page_title="Oil Spill Detection - Real Mask", layout="wide")

st.title("🎯 Oil Spill Detection - Get Correct Mask")
st.write("Get the actual predicted mask from your model")

# First, let's find and fix the model file issue
st.sidebar.header("🔍 Model File Status")

def debug_model_files():
    """Comprehensive model file debugging"""
    st.sidebar.subheader("📁 Current Directory")
    current_dir = os.getcwd()
    st.sidebar.write(f"`{current_dir}`")
    
    all_files = os.listdir('.')
    st.sidebar.subheader("📋 All Files")
    for file in sorted(all_files):
        if os.path.isfile(file):
            size_kb = os.path.getsize(file) / 1024
            st.sidebar.write(f"📄 {file} ({size_kb:.1f} KB)")
        else:
            st.sidebar.write(f"📁 {file}/")
    
    # Look for model files
    model_files = [f for f in all_files if f.endswith(('.pth', '.pt', '.pkl'))]
    
    if model_files:
        st.sidebar.success(f"✅ Found {len(model_files)} model file(s)")
        return model_files
    else:
        st.sidebar.error("❌ No model files found!")
        return []

model_files = debug_model_files()

# Model architecture
class OilSpillModel(nn.Module):
    def __init__(self, num_classes=1):
        super(OilSpillModel, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return torch.sigmoid(x)

@st.cache_resource
def load_real_model():
    """Load the actual model with detailed debugging"""
    if not model_files:
        return None, "No model files found"
    
    for model_file in model_files:
        try:
            st.sidebar.info(f"🔄 Loading: {model_file}")
            
            # Check file size
            file_size = os.path.getsize(model_file) / (1024 * 1024)
            st.sidebar.write(f"📦 File size: {file_size:.1f} MB")
            
            # Try to load
            checkpoint = torch.load(model_file, map_location='cpu')
            st.sidebar.write(f"📊 Loaded type: {type(checkpoint)}")
            
            if isinstance(checkpoint, collections.OrderedDict):
                st.sidebar.write("🔧 Loading state dict...")
                model = OilSpillModel(num_classes=1)
                
                # Try different loading strategies
                try:
                    model.load_state_dict(checkpoint)
                    st.sidebar.success("✅ Exact state dict match!")
                except Exception as e1:
                    st.sidebar.warning(f"⚠️ Exact failed: {str(e1)[:100]}")
                    try:
                        model.load_state_dict(checkpoint, strict=False)
                        st.sidebar.success("✅ Loaded with strict=False")
                    except Exception as e2:
                        st.sidebar.error(f"❌ Strict=False failed: {str(e2)[:100]}")
                        continue
                
                model.eval()
                return model, f"Loaded {model_file}"
                
            elif isinstance(checkpoint, torch.nn.Module):
                st.sidebar.success("✅ Full model loaded!")
                checkpoint.eval()
                return checkpoint, f"Loaded full model: {model_file}"
            else:
                st.sidebar.error(f"❓ Unknown type: {type(checkpoint)}")
                
        except Exception as e:
            st.sidebar.error(f"❌ Load failed: {str(e)}")
            continue
    
    return None, "All loading attempts failed"

# Try to load real model
real_model, load_status = load_real_model()

# Simple demo for testing (only if no real model)
class SimpleDemo:
    def __init__(self):
        self.is_demo = True
    
    def __call__(self, x):
        # Return very low confidence (almost no detection)
        return torch.sigmoid(torch.randn_like(x) * 0.1 - 3.0)  # Very low values

# Use real model if available, otherwise simple demo
if real_model is not None:
    model = real_model
    st.success(f"✅ {load_status}")
else:
    model = SimpleDemo()
    st.warning(f"🔸 DEMO MODE: {load_status}")
    st.info("Using simple demo - upload your model file for real predictions")

# Enhanced preprocessing
def preprocess_for_real_model(image, size=512):
    """Preprocessing that should work with most models"""
    original_size = image.size
    
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image_resized = image.resize((size, size))
    img_array = np.array(image_resized)
    
    # Try multiple normalization strategies
    strategies = []
    
    # Strategy 1: Standard ImageNet (most common)
    img1 = img_array.astype(np.float32) / 255.0
    mean1 = np.array([0.485, 0.456, 0.406])
    std1 = np.array([0.229, 0.224, 0.225])
    img1 = (img1 - mean1) / std1
    strategies.append(('imagenet_standard', img1))
    
    # Strategy 2: Simple 0-1
    img2 = img_array.astype(np.float32) / 255.0
    strategies.append(('simple_0-1', img2))
    
    # Strategy 3: BGR + ImageNet
    img3 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0
    mean3 = np.array([0.406, 0.456, 0.485])  # BGR order
    std3 = np.array([0.225, 0.224, 0.229])
    img3 = (img3 - mean3) / std3
    strategies.append(('bgr_imagenet', img3))
    
    # Strategy 4: Mean subtracted
    img4 = img_array.astype(np.float32) / 255.0
    img4 = (img4 - 0.5) / 0.5
    strategies.append(('mean_subtracted', img4))
    
    return strategies, original_size

def analyze_model_output(output, method_name):
    """Comprehensive analysis of model output"""
    if isinstance(output, (list, tuple)):
        output = output[0]
    
    prob_map = output.squeeze().cpu().numpy()
    
    analysis = {
        'method': method_name,
        'min': float(np.min(prob_map)),
        'max': float(np.max(prob_map)),
        'mean': float(np.mean(prob_map)),
        'std': float(np.std(prob_map)),
        'median': float(np.median(prob_map)),
        'q95': float(np.percentile(prob_map, 95)),
        'above_0.1': float(np.sum(prob_map > 0.1) / prob_map.size * 100),
        'above_0.5': float(np.sum(prob_map > 0.5) / prob_map.size * 100),
        'above_0.8': float(np.sum(prob_map > 0.8) / prob_map.size * 100),
    }
    return analysis, prob_map

# Settings
st.sidebar.header("⚙️ Detection Settings")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.01, 0.99, 0.5, 0.01)
enable_multiple_preprocessing = st.sidebar.checkbox("Try Multiple Preprocessing", value=True)

# Main application
st.header("📡 Upload Image for Real Detection")

uploaded_file = st.file_uploader("Choose image for oil spill detection", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    original_size = image.size
    
    # Display original
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛰️ Original Image")
        st.image(image, use_container_width=True)
        st.write(f"Size: {original_size}")
    
    # Process image
    with st.spinner("🔍 Getting real predictions..."):
        try:
            # Get multiple preprocessing strategies
            preprocessing_strategies, original_size = preprocess_for_real_model(image, size=512)
            
            results = []
            
            for method_name, img_array in preprocessing_strategies:
                # Convert to tensor
                img_tensor = torch.from_numpy(img_array).float()
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                
                # Move to device
                if hasattr(model, 'parameters'):
                    device = next(model.parameters()).device
                    img_tensor = img_tensor.to(device)
                
                # Get prediction
                with torch.no_grad():
                    output = model(img_tensor)
                
                # Analyze output
                analysis, prob_map = analyze_model_output(output, method_name)
                results.append((method_name, analysis, prob_map))
            
            # Find the best preprocessing method
            best_method = None
            best_score = -1
            
            for method_name, analysis, prob_map in results:
                # Score based on reasonable confidence values
                score = 0
                if 0.1 < analysis['mean'] < 0.9:  # Good mean range
                    score += 2
                if analysis['max'] > 0.5:  # Some high confidence
                    score += 1
                if analysis['above_0.1'] > 1:  # Some detection
                    score += 1
                
                if score > best_score:
                    best_score = score
                    best_method = (method_name, analysis, prob_map)
            
            # Use best method or first method
            if best_method:
                method_name, analysis, prob_map = best_method
                st.success(f"🎯 Using best preprocessing: **{method_name}**")
            else:
                method_name, analysis, prob_map = results[0]
                st.info(f"ℹ️ Using preprocessing: **{method_name}**")
            
            # Create mask
            binary_mask = (prob_map > confidence_threshold).astype(np.uint8) * 255
            binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
            
            # Create overlay
            original_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            colored_mask = np.zeros_like(original_cv)
            colored_mask[binary_mask_resized > 0] = [0, 0, 255]  # Red for oil
            blended = cv2.addWeighted(original_cv, 0.7, colored_mask, 0.3, 0)
            blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            
            # Display results
            with col2:
                st.subheader("🎭 Predicted Mask")
                st.image(binary_mask_resized, use_container_width=True)
                
                # Statistics
                oil_pixels = np.sum(binary_mask_resized > 0)
                total_pixels = binary_mask_resized.size
                coverage = (oil_pixels / total_pixels) * 100
                
                st.write(f"**Detection Results:**")
                st.write(f"- Oil Pixels: {oil_pixels:,}")
                st.write(f"- Coverage: {coverage:.4f}%")
                st.write(f"- Threshold: {confidence_threshold}")
                st.write(f"- Preprocessing: {method_name}")
            
            # Model output analysis
            st.subheader("📊 Model Output Analysis")
            
            col_anal1, col_anal2 = st.columns(2)
            
            with col_anal1:
                st.write("**Confidence Statistics:**")
                st.write(f"- Min: {analysis['min']:.6f}")
                st.write(f"- Max: {analysis['max']:.6f}")
                st.write(f"- Mean: {analysis['mean']:.6f}")
                st.write(f"- Std: {analysis['std']:.6f}")
                st.write(f"- Median: {analysis['median']:.6f}")
                st.write(f"- 95th %ile: {analysis['q95']:.6f}")
            
            with col_anal2:
                st.write("**Detection Analysis:**")
                st.write(f"- Pixels > 0.1: {analysis['above_0.1']:.2f}%")
                st.write(f"- Pixels > 0.5: {analysis['above_0.5']:.2f}%")
                st.write(f"- Pixels > 0.8: {analysis['above_0.8']:.2f}%")
                
                # Interpretation
                st.write("**Interpretation:**")
                if analysis['max'] < 0.01:
                    st.error("❌ Model producing very low confidence")
                    st.info("Try different preprocessing or check model")
                elif analysis['mean'] > 0.9:
                    st.warning("⚠️ Model overconfident")
                elif oil_pixels == 0:
                    st.info("ℹ️ No detection - try lower threshold")
                else:
                    st.success("✅ Model producing reasonable output")
            
            # Show all preprocessing results if enabled
            if enable_multiple_preprocessing and len(results) > 1:
                st.subheader("🔍 All Preprocessing Results")
                
                result_cols = st.columns(len(results))
                for idx, (col, (method, analysis, prob_map)) in enumerate(zip(result_cols, results)):
                    with col:
                        test_mask = (prob_map > confidence_threshold).astype(np.uint8) * 255
                        test_mask_resized = cv2.resize(test_mask, (200, 200))
                        
                        st.image(test_mask_resized, use_container_width=True, caption=method)
                        st.write(f"Mean: {analysis['mean']:.4f}")
                        st.write(f"Max: {analysis['max']:.4f}")
            
            # Download results
            st.subheader("💾 Download Real Mask")
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                # Download binary mask
                mask_pil = Image.fromarray(binary_mask_resized)
                buf_mask = io.BytesIO()
                mask_pil.save(buf_mask, format='PNG')
                st.download_button(
                    label="Download Binary Mask",
                    data=buf_mask.getvalue(),
                    file_name="oil_spill_mask.png",
                    mime="image/png"
                )
            
            with col_dl2:
                # Download probability data
                prob_resized = cv2.resize(prob_map, original_size)
                prob_data = {
                    'probabilities': prob_resized,
                    'threshold': confidence_threshold,
                    'method': method_name
                }
                # Save as numpy file
                buf_prob = io.BytesIO()
                np.save(buf_prob, prob_data)
                st.download_button(
                    label="Download Probability Data",
                    data=buf_prob.getvalue(),
                    file_name="oil_spill_probabilities.npy",
                    mime="application/octet-stream"
                )
            
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")

# File upload and troubleshooting
with st.expander("📤 Upload Your Model File"):
    st.markdown("""
    **To get REAL predictions (not demo):**
    """)
    
    uploaded_model = st.file_uploader(
        "Upload your oil_spill_model_deploy.pth file", 
        type=['pth', 'pt', 'pkl']
    )
    
    if uploaded_model is not None:
        try:
            # Save with correct filename
            model_filename = "oil_spill_model_deploy.pth"
            with open(model_filename, "wb") as f:
                f.write(uploaded_model.getvalue())
            
            file_size = os.path.getsize(model_filename) / (1024 * 1024)
            st.success(f"✅ Model uploaded! ({file_size:.1f} MB)")
            st.info("🔄 **Refresh the page** to load your real model")
            
        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")

with st.expander("🔧 Troubleshooting Real Predictions"):
    st.markdown("""
    **If you're not getting correct predictions:**
    
    1. **Check model file:**
       - Ensure `oil_spill_model_deploy.pth` exists
       - File should be in same directory as app.py
       - File size should be reasonable (10-500 MB)
    
    2. **Check model output:**
       - Look at **Model Output Analysis** above
       - If max confidence < 0.01, model might not be working
       - Try different preprocessing methods
    
    3. **Common issues:**
       - **Wrong architecture**: Model doesn't match our defined architecture
       - **Wrong preprocessing**: Model expects different image normalization
       - **Corrupted file**: Model file might be damaged
       - **Wrong task**: Model might be for different type of detection
    
    **Current directory files:**
    """)
    
    files = os.listdir('.')
    for file in sorted(files):
        st.write(f"- `{file}`")
