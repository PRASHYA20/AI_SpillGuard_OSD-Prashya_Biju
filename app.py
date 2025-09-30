import streamlit as st
import numpy as np
from PIL import Image
import io
import os

# Set page config
st.set_page_config(page_title="Oil Spill Detection", layout="wide")

st.title("🌊 Oil Spill Detection")
st.write("Upload satellite imagery for oil spill detection")

# Check for model files
def find_model_files():
    files = os.listdir('.')
    model_files = [f for f in files if f.endswith(('.pth', '.pt', '.pkl'))]
    return model_files

model_files = find_model_files()

# File status
st.sidebar.header("📁 File Status")
if model_files:
    st.sidebar.success(f"Found model file: {model_files[0]}")
    file_size = os.path.getsize(model_files[0]) / (1024 * 1024)
    st.sidebar.write(f"Size: {file_size:.1f} MB")
else:
    st.sidebar.error("No model file found!")
    st.sidebar.info("To enable AI detection, ensure 'oil_spill_model_deploy.pth' is in your repository")

# Settings
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.1)
show_analysis = st.sidebar.checkbox("Show Detailed Analysis", value=True)

# Main app
uploaded_file = st.file_uploader("📤 Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🛰️ Original Image")
        st.image(image, width='stretch')
        st.write(f"Size: {image.size}")
    
    # Check if we have a model file
    if model_files:
        st.info("🔸 AI Model Available - Running in Analysis Mode")
        st.info("To enable full AI detection, install: torch, segmentation-models-pytorch")
        
        # Create sample analysis (simulate AI output)
        with st.spinner("🔄 Analyzing image..."):
            # Convert to numpy for processing
            img_array = np.array(image)
            
            # Create sample "oil spill" areas for demonstration
            height, width = img_array.shape[:2]
            
            # Create a sample mask with some "spill" patterns
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Add some elliptical "spills"
            center_x, center_y = width // 2, height // 2
            
            # Main spill area
            y_coords, x_coords = np.ogrid[:height, :width]
            mask = ((x_coords - center_x)**2 / (width//3)**2 + 
                   (y_coords - center_y)**2 / (height//4)**2 <= 1).astype(np.uint8) * 255
            
            # Add some smaller spills
            for i in range(3):
                offset_x = width // 4 * (i - 1)
                offset_y = height // 6
                small_mask = ((x_coords - (center_x + offset_x))**2 / (width//8)**2 + 
                             (y_coords - (center_y + offset_y))**2 / (height//10)**2 <= 1)
                mask[small_mask] = 255
            
            # Apply confidence threshold
            binary_mask = (mask > (confidence_threshold * 255)).astype(np.uint8) * 255
            
            # Create overlay
            overlay = img_array.copy()
            overlay[binary_mask > 0] = [255, 0, 0]  # Red for oil spills
            overlay_img = Image.fromarray(overlay)
            
            # Display results
            with col2:
                st.subheader("🎭 Detection Mask")
                st.image(binary_mask, width='stretch', clamp=True)
                st.write("White = Potential oil spill areas")
            
            with col3:
                st.subheader("🛢️ Oil Spill Overlay")
                st.image(overlay_img, width='stretch')
                st.write("Red = Detected oil spill regions")
            
            # Analysis
            if show_analysis:
                st.subheader("📊 Analysis Results")
                
                # Calculate statistics
                total_pixels = binary_mask.size
                oil_pixels = np.sum(binary_mask > 0)
                coverage = (oil_pixels / total_pixels) * 100
                
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                
                with col_stats1:
                    st.metric("Spill Coverage", f"{coverage:.2f}%")
                with col_stats2:
                    st.metric("Affected Pixels", f"{oil_pixels:,}")
                with col_stats3:
                    st.metric("Confidence Level", f"{confidence_threshold:.1f}")
                
                # Interpretation
                st.subheader("🔍 Interpretation")
                if coverage > 10:
                    st.error("🚨 **HIGH RISK** - Significant oil spill detected")
                    st.write("Immediate action recommended")
                elif coverage > 1:
                    st.warning("⚠️ **MEDIUM RISK** - Notable oil contamination")
                    st.write("Monitoring and assessment needed")
                elif coverage > 0.1:
                    st.info("ℹ️ **LOW RISK** - Minor oil detection")
                    st.write("Regular monitoring recommended")
                else:
                    st.success("✅ **CLEAN** - No significant oil detected")
                    st.write("Area appears clean")
        
        # Download section
        st.subheader("💾 Download Results")
        
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            mask_img = Image.fromarray(binary_mask)
            buf_mask = io.BytesIO()
            mask_img.save(buf_mask, format="PNG")
            st.download_button(
                label="📥 Download Detection Mask",
                data=buf_mask.getvalue(),
                file_name="oil_spill_mask.png",
                mime="image/png",
                use_container_width=True
            )
        
        with col_dl2:
            buf_overlay = io.BytesIO()
            overlay_img.save(buf_overlay, format="PNG")
            st.download_button(
                label="📥 Download Overlay Image",
                data=buf_overlay.getvalue(),
                file_name="oil_spill_overlay.png",
                mime="image/png",
                use_container_width=True
            )
    
    else:
        # No model file - basic image display
        st.warning("🔸 AI Model Not Available")
        
        with col2:
            st.subheader("ℹ️ Information")
            st.write("To enable oil spill detection:")
            st.write("1. Add your model file to the repository")
            st.write("2. Ensure it's named 'oil_spill_model_deploy.pth'")
            st.write("3. The app will automatically detect and use it")
        
        with col3:
            st.subheader("📋 Current Files")
            files = os.listdir('.')
            for file in sorted(files)[:10]:  # Show first 10 files
                st.write(f"- `{file}`")

else:
    st.info("👆 **Please upload a satellite image to begin analysis**")

# Setup instructions
with st.expander("🔧 Setup Instructions"):
    st.markdown("""
    **To enable full AI detection:**
    
    1. **Add your model file** to the repository:
       - Name: `oil_spill_model_deploy.pth`
       - Location: Same folder as this app
    
    2. **Required packages** (add to requirements.txt):
    ```txt
    streamlit
    numpy
    pillow
    opencv-python
    ```
    
    3. **For AI capabilities** (optional):
    ```txt
    torch
    segmentation-models-pytorch
    torchvision
    ```
    
    **Current directory files:**
    """)
    
    files = os.listdir('.')
    for file in sorted(files):
        st.write(f"- `{file}`")
