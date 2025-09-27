import streamlit as st
import numpy as np
from PIL import Image
import io
import requests
import os

# -----------------------------
# Simple Oil Spill Detection App
# -----------------------------

st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Oil Spill Detection App")
st.write("Upload a satellite image for basic oil spill analysis.")

def simple_spill_detection(image):
    """Simple color-based spill detection"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Simple detection based on dark areas in blue regions
    # Convert to HSV color space
    import cv2
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Define ranges for dark blue areas (potential spills)
    lower_blue = np.array([100, 50, 0])
    upper_blue = np.array([140, 255, 100])
    
    # Create mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    return mask

def basic_color_analysis(image):
    """Basic color analysis without OpenCV"""
    img_array = np.array(image)
    
    # Simple dark area detection
    # Calculate brightness (average of RGB channels)
    brightness = np.mean(img_array, axis=2)
    
    # Dark areas threshold (adjustable)
    dark_threshold = 100
    dark_mask = (brightness < dark_threshold).astype(np.uint8) * 255
    
    return dark_mask

uploaded_file = st.file_uploader("Choose a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open image
        image = Image.open(uploaded_file)
        
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
            st.write(f"Image size: {image.size}")
            st.write(f"Mode: {image.mode}")
        
        with col2:
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Simple analysis - detect dark areas
            st.subheader("Basic Analysis")
            
            # Calculate basic statistics
            avg_brightness = np.mean(img_array)
            min_brightness = np.min(img_array)
            max_brightness = np.max(img_array)
            
            st.write(f"Average brightness: {avg_brightness:.1f}")
            st.write(f"Min brightness: {min_brightness}")
            st.write(f"Max brightness: {max_brightness}")
            
            # Simple dark area detection
            dark_threshold = st.slider("Dark area threshold", 0, 255, 100)
            dark_pixels = np.sum(img_array < dark_threshold) / img_array.size * 100
            
            st.write(f"Dark pixels (< {dark_threshold}): {dark_pixels:.1f}%")
            
            if dark_pixels > 10:
                st.warning("⚠️ Significant dark areas detected - potential spill indicators")
            else:
                st.success("✅ Normal brightness levels detected")
            
            # Create a simple visualization
            try:
                import matplotlib.pyplot as plt
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                
                # Original image
                ax1.imshow(img_array)
                ax1.set_title("Original Image")
                ax1.axis('off')
                
                # Brightness histogram
                brightness = np.mean(img_array, axis=2).flatten()
                ax2.hist(brightness, bins=50, alpha=0.7, color='blue')
                ax2.axvline(dark_threshold, color='red', linestyle='--', label=f'Threshold: {dark_threshold}')
                ax2.set_title("Brightness Distribution")
                ax2.set_xlabel("Brightness")
                ax2.set_ylabel("Pixel Count")
                ax2.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except ImportError:
                st.info("Matplotlib not available for advanced visualization")
        
        # Additional analysis
        st.subheader("Color Channel Analysis")
        
        if len(img_array.shape) == 3:  # Color image
            red_channel = img_array[:, :, 0]
            green_channel = img_array[:, :, 1]
            blue_channel = img_array[:, :, 2]
            
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric("Red Channel Avg", f"{np.mean(red_channel):.1f}")
            with col4:
                st.metric("Green Channel Avg", f"{np.mean(green_channel):.1f}")
            with col5:
                st.metric("Blue Channel Avg", f"{np.mean(blue_channel):.1f}")
        
        # Download analysis report
        report = f"""
        Oil Spill Analysis Report
        =========================
        
        Image Analysis:
        - Size: {image.size}
        - Dark pixels (< {dark_threshold}): {dark_pixels:.1f}%
        - Average brightness: {avg_brightness:.1f}
        
        Assessment:
        - {'Potential spill indicators detected' if dark_pixels > 10 else 'Normal conditions'}
        """
        
        st.download_button(
            label="📄 Download Analysis Report",
            data=report,
            file_name="oil_spill_analysis.txt",
            mime="text/plain"
        )
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("This is a basic version. For advanced detection, ensure all dependencies are installed.")

else:
    st.info("👆 Please upload an image to begin analysis")
    
    # Sample usage instructions
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        1. **Upload a satellite image** of a water body
        2. **Adjust the dark area threshold** to detect potential spills
        3. **Review the analysis results**
        4. **Download the report** if needed
        
        **What to look for:**
        - High percentage of dark pixels in water areas
        - Unusual dark patterns in otherwise bright water
        - Contrast between normal water and potential spill areas
        """)

# Footer
st.markdown("---")
st.markdown("*Note: This is a basic demonstration app. For production use, consider more advanced computer vision techniques.*")
