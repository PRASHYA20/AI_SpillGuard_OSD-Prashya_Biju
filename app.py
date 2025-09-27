import streamlit as st
from PIL import Image
import io

st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Oil Spill Detection App")
st.write("Upload a satellite image for analysis.")

def analyze_image(image):
    """Basic image analysis without NumPy"""
    width, height = image.size
    file_info = {
        "size": f"{width} x {height} pixels",
        "mode": image.mode,
        "format": image.format,
        "file_size": "N/A"
    }
    
    # Basic analysis based on image characteristics
    if image.mode == 'L':
        analysis = "Grayscale image - good for contrast analysis"
    elif image.mode == 'RGB':
        analysis = "Color image - can analyze color channels"
    else:
        analysis = f"Image mode: {image.mode}"
    
    return file_info, analysis

uploaded_file = st.file_uploader("Choose a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open and display image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Basic file info
            file_size = len(uploaded_file.getvalue()) / 1024  # KB
            st.write(f"File size: {file_size:.1f} KB")
            st.write(f"File type: {uploaded_file.type}")
        
        with col2:
            # Image analysis
            file_info, analysis = analyze_image(image)
            
            st.subheader("Image Analysis")
            st.write(f"Dimensions: {file_info['size']}")
            st.write(f"Color mode: {file_info['mode']}")
            st.write(f"Format: {file_info['format']}")
            st.write(f"Assessment: {analysis}")
            
            # Simple spill detection based on image characteristics
            width, height = image.size
            if width > 1000 and height > 1000:
                st.success("✅ High-resolution image - good for analysis")
            else:
                st.warning("⚠️ Lower resolution - consider higher quality images")
            
            if image.mode == 'RGB':
                st.info("🔍 Color image - can detect color anomalies")
            else:
                st.info("⚫ Grayscale image - analyzing contrast patterns")
        
        # Advanced analysis section (if NumPy is available)
        try:
            import numpy as np
            st.subheader("Advanced Analysis")
            
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Basic statistics
            if len(img_array.shape) == 3:  # Color image
                avg_brightness = np.mean(img_array)
                st.write(f"Average brightness: {avg_brightness:.1f}")
                
                # Simple dark area detection
                dark_threshold = st.slider("Dark area threshold", 0, 255, 100)
                dark_pixels = np.sum(img_array < dark_threshold) / img_array.size * 100
                st.write(f"Dark pixels (< {dark_threshold}): {dark_pixels:.1f}%")
                
                if dark_pixels > 15:
                    st.error("🚨 High percentage of dark areas - potential spill indicator")
                elif dark_pixels > 5:
                    st.warning("⚠️ Moderate dark areas detected")
                else:
                    st.success("✅ Normal brightness levels")
            
        except ImportError:
            st.info("NumPy not available for advanced analysis")
        
        # Download options
        st.subheader("Export Options")
        
        # Convert to RGB if needed for download
        if image.mode != 'RGB':
            download_image = image.convert('RGB')
        else:
            download_image = image
            
        # Download processed image
        buf = io.BytesIO()
        download_image.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        
        st.download_button(
            label="💾 Download Processed Image",
            data=buf,
            file_name="processed_image.jpg",
            mime="image/jpeg",
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("This is a basic version for demonstration.")

else:
    st.info("👆 Please upload a satellite image to begin analysis")
    
    # Sample instructions
    with st.expander("📋 How to use this app"):
        st.markdown("""
        **Instructions:**
        1. Upload a satellite image of a water body
        2. View basic image information and analysis
        3. Adjust settings for dark area detection
        4. Download processed results
        
        **Recommended images:**
        - Satellite images of oceans, seas, or large lakes
        - Images with good contrast and resolution
        - Both color and grayscale images work
        
        **What this app detects:**
        - Image characteristics and metadata
        - Basic brightness patterns
        - Potential dark areas that might indicate spills
        """)

st.markdown("---")
st.markdown("*Basic oil spill detection demonstration*")
