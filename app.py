import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests
import os
import io
import cv2
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# -----------------------------
# Dropbox Model URL (we'll use a simpler approach)
# -----------------------------
MODEL_PATH = "oil_spill_model_deploy.pth"

# -----------------------------
# Oil Spill Detection using traditional CV + ML
# -----------------------------
class OilSpillDetector:
    def __init__(self):
        self.detector = IsolationForest(contamination=0.1, random_state=42)
        
    def extract_features(self, image_array):
        """Extract features from image for spill detection"""
        features = []
        
        # Color features (RGB, HSV)
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # Mean and std of each channel
        for channel in range(3):
            features.extend([np.mean(image_array[:,:,channel]), np.std(image_array[:,:,channel])])
            features.extend([np.mean(hsv[:,:,channel]), np.std(hsv[:,:,channel])])
        
        # Texture features (gradient)
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features.extend([np.mean(gradient_magnitude), np.std(gradient_magnitude)])
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        features.append(np.sum(edges > 0) / edges.size)
        
        return np.array(features)
    
    def detect_spills(self, image):
        """Detect oil spills using traditional computer vision"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Resize for processing
        img_resized = cv2.resize(img_array, (256, 256))
        
        # Method 1: Color-based segmentation (oil spills often have dark, smooth areas)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
        
        # Define range for dark areas (potential oil spills)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 100])
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # Method 2: Texture-based detection (oil spills have smooth texture)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        
        # Calculate local variance (smooth areas have low variance)
        kernel = np.ones((15, 15), np.float32) / 225
        smoothed = cv2.filter2D(gray, -1, kernel)
        variance = cv2.filter2D(gray**2, -1, kernel) - smoothed**2
        smooth_mask = (variance < 100).astype(np.uint8) * 255
        
        # Combine masks
        combined_mask = cv2.bitwise_or(dark_mask, smooth_mask)
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Oil Spill Detection (Computer Vision)")
st.write("Upload a satellite image to detect possible oil spills using computer vision techniques.")

# Initialize detector
if 'detector' not in st.session_state:
    st.session_state.detector = OilSpillDetector()

# Sidebar with settings
with st.sidebar:
    st.header("Detection Settings")
    
    detection_method = st.selectbox(
        "Detection Method",
        ["Color + Texture", "Color-Based", "Texture-Based"],
        help="Choose the detection algorithm"
    )
    
    sensitivity = st.slider(
        "Sensitivity",
        min_value=1,
        max_value=10,
        value=5,
        help="Higher values detect more potential spills"
    )
    
    min_spill_size = st.slider(
        "Minimum Spill Size (pixels)",
        min_value=10,
        max_value=1000,
        value=100,
        help="Filter out small detected areas"
    )

uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
        st.write(f"Image size: {image.size}")
    
    with col2:
        with st.spinner("Analyzing image for oil spills..."):
            # Convert to numpy array
            img_array = np.array(image)
            
            # Detect spills
            detector = st.session_state.detector
            
            if detection_method == "Color + Texture":
                mask = detector.detect_spills(image)
            elif detection_method == "Color-Based":
                hsv = cv2.cvtColor(cv2.resize(img_array, (256, 256)), cv2.COLOR_RGB2HSV)
                lower_dark = np.array([0, 0, 0])
                upper_dark = np.array([180, 255, 100 + sensitivity * 10])
                mask = cv2.inRange(hsv, lower_dark, upper_dark)
            else:  # Texture-Based
                gray = cv2.cvtColor(cv2.resize(img_array, (256, 256)), cv2.COLOR_RGB2GRAY)
                kernel = np.ones((15, 15), np.float32) / 225
                smoothed = cv2.filter2D(gray, -1, kernel)
                variance = cv2.filter2D(gray**2, -1, kernel) - smoothed**2
                mask = (variance < 50 + sensitivity * 20).astype(np.uint8) * 255
            
            # Filter small areas
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_mask = np.zeros_like(mask)
            for contour in contours:
                if cv2.contourArea(contour) > min_spill_size:
                    cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
            
            # Create visualization
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            ax1.imshow(image)
            ax1.set_title("Original Image")
            ax1.axis("off")
            
            # Detection mask
            ax2.imshow(filtered_mask, cmap="hot")
            ax2.set_title("Detection Mask")
            ax2.axis("off")
            
            # Overlay
            overlay = cv2.resize(img_array, (256, 256))
            ax3.imshow(overlay)
            ax3.imshow(filtered_mask, cmap="Reds", alpha=0.5)
            ax3.set_title("Overlay (Red = Potential Spill)")
            ax3.axis("off")
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Statistics
    spill_area = np.sum(filtered_mask > 0) / (filtered_mask.shape[0] * filtered_mask.shape[1]) * 100
    num_spills = len(contours)
    
    st.subheader("📊 Detection Results")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Spill Area Percentage", f"{spill_area:.2f}%")
    
    with col4:
        st.metric("Number of Detected Areas", num_spills)
    
    with col5:
        status = "🟢 No Significant Spills" if spill_area < 1.0 else "🔴 Potential Spills Detected"
        st.metric("Status", status)
    
    # Download results
    if spill_area > 0:
        mask_img = Image.fromarray(filtered_mask)
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="💾 Download Detection Mask",
            data=byte_im,
            file_name="oil_spill_detection.png",
            mime="image/png",
            use_container_width=True
        )

else:
    st.info("👆 Please upload a satellite image to get started.")
    st.markdown("""
    ### Sample images to test:
    - Dark, smooth areas in water bodies
    - Satellite images of oceans, seas, or large lakes
    - Images with potential oil spill patterns
    """)

# Footer
st.markdown("---")
st.markdown("### How it works:")
st.markdown("""
This application uses computer vision techniques to detect potential oil spills:

1. **Color Analysis**: Detects dark areas typical of oil spills
2. **Texture Analysis**: Identifies smooth surface patterns
3. **Morphological Operations**: Cleans up detection results
4. **Size Filtering**: Removes small false positives

**Note**: This is a demonstration using traditional computer vision. For production use, consider training a dedicated ML model.
""")
