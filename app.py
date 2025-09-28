import streamlit as st
import traceback

try:
    from PIL import Image
    import numpy as np
    import requests
    import os

    # -----------------------------
    # Dropbox Model URL
    # -----------------------------
    MODEL_PATH = "oil_spill_model_deploy.pth"
    DROPBOX_URL = "https://www.dropbox.com/scl/fi/stl47n6ixrzv59xs2jt4m/oil_spill_model_deploy.pth?rlkey=rojyk0fq73mk8tai8jc3exrev&dl=1"

    # -----------------------------
    # Streamlit App
    # -----------------------------
    st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊", layout="wide")
    st.title("🌊 Oil Spill Detection App")
    st.success("✅ App loaded successfully!")

    with st.sidebar:
        st.header("⚙️ Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.1)
        st.header("ℹ️ About")
        st.write("Upload satellite images for oil spill detection.")

    # File uploader
    uploaded_file = st.file_uploader("📤 Upload Satellite Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Load and display image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.success("✅ Image loaded successfully!")
            
            # Simple analysis without complex processing
            if st.button("Analyze Image"):
                try:
                    with st.spinner("Analyzing image..."):
                        # Very basic image analysis - no ML, no complex operations
                        img_array = np.array(image)
                        
                        # Display basic info
                        st.write(f"**Image Dimensions:** {img_array.shape}")
                        st.write(f"**Image Size:** {img_array.shape[0]} x {img_array.shape[1]} pixels")
                        
                        # Simple color analysis
                        avg_color = np.mean(img_array, axis=(0, 1))
                        st.write(f"**Average Color (RGB):** {avg_color.astype(int)}")
                        
                        st.success("✅ Analysis complete!")
                        st.info("ML features can be added once the app is stable")
                        
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    st.info("This is a basic analysis. ML features will be added later.")
                    
        except Exception as e:
            st.error(f"❌ Error loading image: {e}")
            st.info("Try a different image file")

    else:
        st.info("👆 Please upload a satellite image")

    # Simple model download status (no processing)
    if st.checkbox("Show model download status"):
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
            st.success(f"✅ Model file available ({file_size:.1f} MB)")
        else:
            st.warning("⚠️ Model file not downloaded yet")
            if st.button("Download Model File"):
                try:
                    st.info("Starting download...")
                    response = requests.get(DROPBOX_URL, stream=True, timeout=30)
                    response.raise_for_status()
                    
                    # Simple download without progress bar
                    with open(MODEL_PATH, 'wb') as f:
                        f.write(response.content)
                    
                    st.success("✅ Model downloaded successfully!")
                except Exception as e:
                    st.error(f"❌ Download failed: {e}")

except Exception as e:
    st.error(f"🚨 App failed to start: {e}")
    st.code(traceback.format_exc())
