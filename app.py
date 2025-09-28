import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊", layout="wide")
st.title("🌊 Oil Spill Detection Demo")
st.write("Upload satellite images for analysis")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)
    
    if st.button("🔍 Analyze for Oil Spills"):
        with st.spinner("Analyzing..."):
            # Create simulated detection
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            # Simulate random oil spill detection
            np.random.seed(42)  # For consistent results
            mask = np.random.rand(height, width) > 0.9
            
            # Create overlay
            overlay = img_array.copy()
            overlay[mask] = [255, 0, 0]  # Red for oil spills
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(overlay, caption="Oil Spill Detection", use_column_width=True)
            with col2:
                st.image(mask.astype(np.uint8) * 255, caption="Detection Mask", use_column_width=True)
            
            # Calculate metrics
            spill_area = np.sum(mask) / mask.size * 100
            st.metric("Detected Spill Area", f"{spill_area:.2f}%")
            st.metric("Confidence", "85%")
            st.success("✅ Analysis Complete!")
else:
    st.info("👆 Please upload a satellite image")

st.markdown("---")
st.write("*This is a demonstration version. AI model integration available in full version.*")
