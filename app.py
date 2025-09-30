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
st.set_page_config(page_title="Oil Spill Detection", layout="wide")

st.title("🛢️ Oil Spill Detection")
st.write("Get the correct predicted mask from your model")

# Simple model architecture
class SimpleOilSpillModel(nn.Module):
    def __init__(self):
        super(SimpleOilSpillModel, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(1024, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.final = nn.Conv2d(64, 1, 1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        return torch.sigmoid(x)

# Check for model files
def find_model_files():
    files = os.listdir('.')
    model_files = [f for f in files if f.endswith(('.pth', '.pt', '.pkl'))]
    return model_files

model_files = find_model_files()

# File status
st.sidebar.header("📁 File Status")
if model_files:
    st.sidebar.success(f"Found {len(model_files)} model files:")
    for file in model_files:
        size_mb = os.path.getsize(file) / (1024 * 1024)
        st.sidebar.write(f"• {file} ({size_mb:.1f} MB)")
else:
    st.sidebar.error("No model files found!")

# Try to load model
model = None
if model_files:
    try:
        # Try the first model file
        model_file = model_files[0]
        st.sidebar.info(f"Loading {model_file}...")
        
        checkpoint = torch.load(model_file, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            # It's a state dict
            model = SimpleOilSpillModel()
            try:
                model.load_state_dict(checkpoint, strict=False)
                st.sidebar.success("✅ Model loaded with strict=False")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load: {e}")
                model = None
        else:
            # It's a full model
            model = checkpoint
            st.sidebar.success("✅ Full model loaded")
            
        if model is not None:
            model.eval()
            
    except Exception as e:
        st.sidebar.error(f"❌ Load error: {e}")
        model = None

# Demo model as fallback
class DemoModel:
    def __init__(self):
        self.is_demo = True
    
    def __call__(self, x):
        # Return very low confidence (almost no detection)
        shape = x.shape
        return torch.sigmoid(torch.randn(shape[0], 1, shape[2]//4, shape[3]//4) * 0.1 - 3.0)

if model is None:
    st.warning("🔸 **Running in Demo Mode** - No model loaded")
    st.info("Upload your model file below to get real predictions")
    model = DemoModel()
else:
    st.success("✅ **Real Model Loaded** - Showing actual predictions")

# Simple preprocessing
def preprocess_image(image, size=512):
    original_size = image.size
    
    # Convert to RGB and resize
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((size, size))
    
    # Convert to numpy and normalize
    img_array = np.array(image) / 255.0
    
    # Try different normalization methods
    methods = []
    
    # Method 1: ImageNet standard
    mean1 = np.array([0.485, 0.456, 0.406])
    std1 = np.array([0.229, 0.224, 0.225])
    img1 = (img_array - mean1) / std1
    methods.append(('imagenet', img1))
    
    # Method 2: Simple 0-1
    img2 = img_array.copy()
    methods.append(('0-1_normalized', img2))
    
    # Method 3: BGR + ImageNet
    img3 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) / 255.0
    mean3 = np.array([0.406, 0.456, 0.485])
    std3 = np.array([0.225, 0.224, 0.229])
    img3 = (img3 - mean3) / std3
    methods.append(('bgr_imagenet', img3))
    
    return methods, original_size

# Settings
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Confidence Threshold", 0.01, 0.99, 0.5, 0.01)
input_size = st.sidebar.selectbox("Input Size", [256, 384, 512], index=2)

# Main app
st.header("📡 Upload Image")

uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        st.write(f"Size: {image.size}")
    
    # Process image
    with st.spinner("Processing..."):
        try:
            # Get preprocessing methods
            methods, original_size = preprocess_image(image, size=input_size)
            
            # Try each preprocessing method
            all_results = []
            for method_name, img_array in methods:
                # Convert to tensor
                img_tensor = torch.from_numpy(img_array).float()
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                
                # Predict
                with torch.no_grad():
                    output = model(img_tensor)
                
                if isinstance(output, (list, tuple)):
                    output = output[0]
                
                # Get probability map
                prob_map = output.squeeze().cpu().numpy()
                
                # Create mask
                mask = (prob_map > threshold).astype(np.uint8) * 255
                mask_resized = cv2.resize(mask, original_size)
                
                # Statistics
                oil_pixels = np.sum(mask_resized > 0)
                total_pixels = mask_resized.size
                coverage = (oil_pixels / total_pixels) * 100
                
                all_results.append({
                    'method': method_name,
                    'mask': mask_resized,
                    'prob_map': prob_map,
                    'oil_pixels': oil_pixels,
                    'coverage': coverage,
                    'max_confidence': np.max(prob_map),
                    'mean_confidence': np.mean(prob_map)
                })
            
            # Find best result (most reasonable detection)
            best_result = None
            for result in all_results:
                if result['max_confidence'] > 0.1 and result['mean_confidence'] < 0.9:
                    best_result = result
                    break
            
            if best_result is None:
                # Use the first result
                best_result = all_results[0]
            
            # Display results
            with col2:
                st.subheader("Predicted Mask")
                st.image(best_result['mask'], use_container_width=True)
                
                st.write("**Detection Results:**")
                st.write(f"- Oil Pixels: {best_result['oil_pixels']:,}")
                st.write(f"- Coverage: {best_result['coverage']:.4f}%")
                st.write(f"- Max Confidence: {best_result['max_confidence']:.4f}")
                st.write(f"- Method: {best_result['method']}")
            
            with col3:
                st.subheader("Overlay")
                # Create overlay
                original_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                colored = np.zeros_like(original_cv)
                colored[best_result['mask'] > 0] = [0, 0, 255]  # Red
                blended = cv2.addWeighted(original_cv, 0.7, colored, 0.3, 0)
                blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
                
                st.image(blended_rgb, use_container_width=True)
                st.write(f"Threshold: {threshold}")
            
            # Model analysis
            st.subheader("🔍 Model Analysis")
            
            col_anal1, col_anal2 = st.columns(2)
            
            with col_anal1:
                st.write("**All Preprocessing Methods:**")
                for result in all_results:
                    status = "🎯" if result == best_result else "•"
                    st.write(f"{status} {result['method']}: "
                           f"max={result['max_confidence']:.4f}, "
                           f"coverage={result['coverage']:.4f}%")
            
            with col_anal2:
                st.write("**Interpretation:**")
                if best_result['max_confidence'] < 0.01:
                    st.error("Model producing very low confidence")
                    st.info("This might indicate:")
                    st.info("• Wrong model architecture")
                    st.info("• Wrong preprocessing")
                    st.info("• Model not trained properly")
                elif best_result['oil_pixels'] == 0:
                    st.warning("No oil detected")
                    st.info("Try lowering the threshold")
                else:
                    st.success("Model is producing detections")
            
            # Download
            st.subheader("💾 Download")
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                mask_pil = Image.fromarray(best_result['mask'])
                buf = io.BytesIO()
                mask_pil.save(buf, format='PNG')
                st.download_button(
                    label="Download Mask",
                    data=buf.getvalue(),
                    file_name="oil_spill_mask.png",
                    mime="image/png"
                )
            
            with col_dl2:
                # Save probabilities
                prob_data = {
                    'probabilities': best_result['prob_map'],
                    'threshold': threshold,
                    'method': best_result['method']
                }
                buf_prob = io.BytesIO()
                np.save(buf_prob, prob_data, allow_pickle=True)
                st.download_button(
                    label="Download Probabilities",
                    data=buf_prob.getvalue(),
                    file_name="probabilities.npy",
                    mime="application/octet-stream"
                )
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# File upload section
with st.expander("📤 Upload Model File"):
    st.markdown("""
    **To use your real model:**
    1. Upload your `.pth` or `.pt` file
    2. Refresh the page
    3. The app will automatically load it
    """)
    
    uploaded_model = st.file_uploader(
        "Choose model file",
        type=['pth', 'pt', 'pkl'],
        key="model_upload"
    )
    
    if uploaded_model is not None:
        try:
            # Save the file
            filename = "uploaded_model.pth"
            with open(filename, "wb") as f:
                f.write(uploaded_model.getvalue())
            
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            st.success(f"✅ Model uploaded! ({size_mb:.1f} MB)")
            st.info("🔄 **Refresh the page** to load your model")
            
        except Exception as e:
            st.error(f"Upload failed: {e}")

# Quick help
with st.expander("❓ Help"):
    st.markdown("""
    **Common Issues:**
    
    - **No model file**: Upload your `.pth` file using the section above
    - **Wrong predictions**: Try different preprocessing methods
    - **Low confidence**: Model might not be working correctly
    
    **Expected Model File:**
    - Name: `oil_spill_model_deploy.pth`
    - Type: PyTorch model file (.pth, .pt, .pkl)
    - Location: Same folder as this app
    
    **Current files:**
    """)
    
    files = os.listdir('.')
    for file in sorted(files):
        st.write(f"- `{file}`")
