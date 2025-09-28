# app_streamlit.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import requests
import torch
import torch.nn as nn
from torchvision import transforms
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Oil Spill Detection System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ff4b4b;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #ffa500;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .risk-low {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Define the model architecture
class OilSpillModel(nn.Module):
    def __init__(self, num_classes=1):
        super(OilSpillModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 32 * 32, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

@st.cache_resource
def download_model_from_dropbox():
    """Download the model from Dropbox with proper error handling"""
    # Try different Dropbox URL formats
    dropbox_urls = [
        "https://www.dropbox.com/scl/fi/stl47n6ixrzv59xs2jt4m/oil_spill_model_deploy.pth?rlkey=rojyk0fq73mk8tai8jc3exrev&dl=1",
        "https://dl.dropbox.com/scl/fi/stl47n6ixrzv59xs2jt4m/oil_spill_model_deploy.pth?rlkey=rojyk0fq73mk8tai8jc3exrev&st=l1zhnigc&dl=1",
        "https://www.dropbox.com/scl/fi/stl47n6ixrzv59xs2jt4m/oil_spill_model_deploy.pth?rlkey=rojyk0fq73mk8tai8jc3exrev&raw=1"
    ]
    
    for i, dropbox_url in enumerate(dropbox_urls):
        try:
            st.info(f"📥 Attempting to download model (Attempt {i+1}/3)...")
            
            # Use a session with headers to mimic browser
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            response = session.get(dropbox_url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Check if we got a valid file (not HTML error page)
            if len(response.content) < 1000 or b"html" in response.content[:1000].lower():
                st.warning(f"⚠️ Attempt {i+1} failed: Got HTML instead of model file")
                continue
            
            model_path = "oil_spill_model_deploy.pth"
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify file was downloaded properly
            if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:
                st.success(f"✅ Model downloaded successfully! (Size: {os.path.getsize(model_path)} bytes)")
                return model_path
            else:
                st.warning(f"⚠️ Attempt {i+1} failed: File too small or corrupted")
                
        except Exception as e:
            st.warning(f"⚠️ Attempt {i+1} failed: {str(e)}")
            continue
    
    st.error("❌ All download attempts failed. Using computer vision only.")
    return None

@st.cache_resource
def load_oil_spill_model():
    """Load the PyTorch model with fallback"""
    try:
        # Try to download model
        model_path = download_model_from_dropbox()
        
        if model_path and os.path.exists(model_path):
            # Initialize model
            model = OilSpillModel(num_classes=1)
            
            # Load state dict
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            st.success("✅ PyTorch model loaded successfully!")
            return model
        else:
            st.warning("⚠️ No model file available. Using computer vision only.")
            return None
            
    except Exception as e:
        st.warning(f"⚠️ Model loading failed: {str(e)}. Using computer vision only.")
        return None

def load_and_preprocess_image(image):
    """Load and preprocess image"""
    if isinstance(image, str):
        # If it's a file path
        img = Image.open(image).convert('RGB')
    else:
        # If it's an uploaded file
        img = Image.open(image).convert('RGB')
    
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_bgr, img_array

def identify_water_regions(hsv, lab, gray):
    """Identify water regions while excluding land and other non-water areas"""
    # Water typically has blue-green hues in HSV
    water_low1 = np.array([90, 40, 40])
    water_high1 = np.array([130, 255, 200])
    
    water_low2 = np.array([70, 30, 50])
    water_high2 = np.array([100, 200, 180])
    
    water_mask1 = cv2.inRange(hsv, water_low1, water_high1)
    water_mask2 = cv2.inRange(hsv, water_low2, water_high2)
    
    water_mask = cv2.bitwise_or(water_mask1, water_mask2)
    
    # Exclude very bright areas (clouds, waves, sun glare)
    brightness_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
    water_mask = cv2.bitwise_and(water_mask, cv2.bitwise_not(brightness_mask))
    
    # Exclude very dark areas (shadows, land)
    darkness_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 30]))
    water_mask = cv2.bitwise_and(water_mask, cv2.bitwise_not(darkness_mask))
    
    # Clean up the water mask
    kernel = np.ones((5, 5), np.uint8)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
    
    # Remove small water regions
    contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            cv2.drawContours(water_mask, [contour], 0, 0, -1)
    
    return water_mask

def calculate_texture_variance(gray):
    """Calculate texture variance using local standard deviation"""
    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    
    mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    variance = cv2.filter2D((gray.astype(np.float32) - mean) ** 2, -1, kernel)
    texture_variance = np.sqrt(variance).astype(np.uint8)
    
    return texture_variance

def detect_oil_spills_accurate(image):
    """
    Accurate oil spill detection using computer vision
    """
    img_bgr, img_rgb = load_and_preprocess_image(image)
    original_img = img_bgr.copy()
    
    # Resize for consistent processing
    target_size = (512, 512)
    img_resized = cv2.resize(img_bgr, target_size)
    original_img_resized = cv2.resize(original_img, target_size)
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Identify water regions
    water_mask = identify_water_regions(hsv, lab, gray)
    
    # Step 2: Detect oil candidates in water
    oil_candidates = np.zeros_like(gray)
    
    # Dark patches in water (crude oil)
    dark_water = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
    dark_in_water = cv2.bitwise_and(dark_water, water_mask)
    
    # Smooth texture areas
    texture_map = calculate_texture_variance(gray)
    smooth_areas = cv2.inRange(texture_map, 0, 20)
    smooth_in_water = cv2.bitwise_and(smooth_areas, water_mask)
    
    # Combine detection methods
    oil_candidates = cv2.bitwise_or(dark_in_water, smooth_in_water)
    oil_candidates = cv2.bitwise_and(oil_candidates, water_mask)
    
    # Clean up oil mask
    kernel = np.ones((5, 5), np.uint8)
    oil_mask = cv2.morphologyEx(oil_candidates, cv2.MORPH_CLOSE, kernel)
    oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_OPEN, kernel)
    
    # Remove small regions
    contours, _ = cv2.findContours(oil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            cv2.drawContours(oil_mask, [contour], 0, 0, -1)
    
    # Validate oil regions
    validated_mask = oil_mask.copy()
    oil_properties = []
    
    contours, _ = cv2.findContours(validated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Oil spills often have irregular shapes
        is_likely_oil = circularity < 0.7 and area > 100
        
        oil_properties.append({
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'is_likely_oil': is_likely_oil
        })
        
        if not is_likely_oil:
            cv2.drawContours(validated_mask, [contour], 0, 0, -1)
    
    # Create result visualization
    result_img = original_img_resized.copy()
    
    # Water in blue (semi-transparent)
    water_overlay = result_img.copy()
    water_overlay[water_mask > 0] = [255, 0, 0]  # Blue in BGR
    cv2.addWeighted(water_overlay, 0.2, result_img, 0.8, 0, result_img)
    
    # Oil spills in red (semi-transparent)
    oil_overlay = result_img.copy()
    oil_overlay[validated_mask > 0] = [0, 0, 255]  # Red in BGR
    cv2.addWeighted(oil_overlay, 0.4, result_img, 0.6, 0, result_img)
    
    # Draw contours for oil spills
    contours, _ = cv2.findContours(validated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_img, contours, -1, (0, 0, 255), 2)
    
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    # Calculate statistics
    water_pixels = np.sum(water_mask > 0)
    oil_pixels = np.sum(validated_mask > 0)
    oil_percentage = (oil_pixels / water_pixels * 100) if water_pixels > 0 else 0
    
    contour_count = len([c for c in contours if cv2.contourArea(c) > 50])
    
    return result_img, oil_percentage, contour_count, validated_mask, cv2.cvtColor(original_img_resized, cv2.COLOR_BGR2RGB), oil_properties, water_mask

def main():
    st.markdown('<h1 class="main-header">🌊 Oil Spill Detection System</h1>', unsafe_allow_html=True)
    
    # Show app status
    with st.sidebar:
        st.title("App Status")
        if st.button("Check Model Status"):
            model = load_oil_spill_model()
            if model:
                st.success("✅ AI Model Ready")
            else:
                st.info("🔍 Using Computer Vision")
    
    st.info("💡 **Tip**: Upload satellite images of ocean areas for best results")
    
    uploaded_file = st.file_uploader("Choose a satellite image", type=['png', 'jpg', 'jpeg', 'tiff'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption="Uploaded Satellite Image")
        
        if st.button("🔍 Analyze for Oil Spills", type="primary"):
            with st.spinner("Analyzing image for oil spills..."):
                try:
                    # Use detection functions
                    result_img, oil_percentage, contour_count, oil_mask, original_img, oil_properties, water_mask = detect_oil_spills_accurate(uploaded_file)
                    
                    # Display results
                    with col2:
                        st.subheader("Detection Results")
                        st.image(result_img, use_column_width=True, caption="Oil Spill Detection Results")
                    
                    # Show statistics
                    st.markdown("---")
                    st.subheader("📊 Detection Statistics")
                    
                    col3, col4, col5, col6 = st.columns(4)
                    
                    with col3:
                        if oil_percentage > 1.0:
                            st.markdown('<div class="risk-high">HIGH RISK</div>', unsafe_allow_html=True)
                        elif oil_percentage > 0.1:
                            st.markdown('<div class="risk-medium">MEDIUM RISK</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="risk-low">LOW RISK</div>', unsafe_allow_html=True)
                        
                        st.metric("Risk Level", 
                                 "HIGH" if oil_percentage > 1.0 else "MEDIUM" if oil_percentage > 0.1 else "LOW")
                    
                    with col4:
                        st.metric("Oil Spill Area", f"{oil_percentage:.4f}%")
                    
                    with col5:
                        st.metric("Detected Regions", contour_count)
                    
                    with col6:
                        likely_oil_count = sum(1 for prop in oil_properties if prop['is_likely_oil'])
                        st.metric("Confirmed Spills", likely_oil_count)
                    
                    # Detailed analysis
                    with st.expander("📋 Detailed Analysis Report"):
                        st.write(f"**Detection Method**: Advanced Computer Vision")
                        st.write(f"- **Total Water Area**: {np.sum(water_mask > 0):,} pixels")
                        st.write(f"- **Oil Spill Area**: {np.sum(oil_mask > 0):,} pixels")
                        st.write(f"- **Oil Coverage**: {oil_percentage:.4f}% of water area")
                        
                        if oil_percentage > 1.0:
                            st.error("🚨 **HIGH RISK**: Significant oil spill detected! Immediate attention required.")
                        elif oil_percentage > 0.1:
                            st.warning("⚠️ **MEDIUM RISK**: Potential oil contamination detected. Monitor closely.")
                        else:
                            st.success("✅ **LOW RISK**: No significant oil spills detected.")
                
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
                    st.info("💡 Try with a different image or check if it's a valid satellite image.")

if __name__ == "__main__":
    main()
