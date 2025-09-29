import streamlit as st
import os
import torch
import numpy as np
import cv2
from PIL import Image
import io

# Set page config
st.set_page_config(page_title="U-Net Debug", layout="wide")

st.title("🐛 U-Net Model Debugger")
st.write("Let's find your model file and fix the loading issue")

# Debug: List ALL files in current directory
st.header("📁 Files in Your Repository")
try:
    all_files = os.listdir('.')
    st.write("**All files:**", all_files)
    
    # Filter for model files
    model_extensions = ('.pth', '.pt', '.pkl', '.h5', '.keras', '.onnx', '.bin')
    model_files = [f for f in all_files if f.endswith(model_extensions)]
    
    st.write("**Potential model files:**", model_files if model_files else "No model files found!")
    
except Exception as e:
    st.error(f"Error listing files: {e}")

# Try to load ANY model file we find
@st.cache_resource
def try_load_any_model():
    st.header("🔍 Attempting to Load Model Files")
    
    all_files = os.listdir('.')
    model_extensions = ('.pth', '.pt', '.pkl', '.h5', '.keras', '.onnx')
    
    for file in all_files:
        if file.endswith(model_extensions):
            st.write(f"--- Trying to load: `{file}` ---")
            
            try:
                if file.endswith(('.pth', '.pt')):
                    # Try PyTorch loading
                    if torch.cuda.is_available():
                        checkpoint = torch.load(file, map_location='cuda')
                    else:
                        checkpoint = torch.load(file, map_location='cpu')
                    
                    st.success(f"✅ SUCCESS: Loaded {file} as PyTorch file!")
                    
                    # Analyze the checkpoint structure
                    st.write("**Checkpoint structure:**", type(checkpoint))
                    
                    if isinstance(checkpoint, dict):
                        st.write("**Checkpoint keys:**", list(checkpoint.keys()))
                        
                        # Show some values
                        for key, value in checkpoint.items():
                            if hasattr(value, 'shape'):
                                st.write(f"  {key}: shape {value.shape}")
                            else:
                                st.write(f"  {key}: {type(value)}")
                    
                    return checkpoint, file
                    
            except Exception as e:
                st.error(f"❌ FAILED to load {file}: {str(e)}")
                continue
    
    st.error("❌ No model files could be loaded")
    return None, None

# Try loading
checkpoint, loaded_file = try_load_any_model()

# Display results
st.header("📊 Loading Results")
if checkpoint is not None:
    st.success(f"🎉 Successfully loaded: `{loaded_file}`")
    
    # Show checkpoint details
    if isinstance(checkpoint, dict):
        st.subheader("📋 Checkpoint Contents:")
        for key, value in checkpoint.items():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(f"**{key}:**")
            with col2:
                if hasattr(value, 'shape'):
                    st.write(f"Shape: {value.shape}")
                elif isinstance(value, (int, float, str)):
                    st.write(f"Value: {value}")
                else:
                    st.write(f"Type: {type(value)}")
else:
    st.error("🚨 No model files could be loaded")

# Instructions based on findings
st.header("🔧 Next Steps")

if checkpoint is None:
    st.markdown("""
    ### ❌ No model files found or loaded
    
    **Please:**
    1. **Ensure your model file is in the same directory as `app.py`**
    2. **Common model file names:**
       - `model.pth`, `model.pt`, `unet.pth`, `best_model.pth`
       - `checkpoint.pth`, `weights.pth`, `trained_model.pth`
    
    3. **If your model file has a different name**, rename it to one of the above
    4. **Make sure the file is committed to Git**
    
    **Check your repository structure:**
    ```
    your-repo/
    ├── app.py
    ├── requirements.txt
    ├── your-model-file.pth  # ← This should exist!
    └── other files...
    ```
    """)
else:
    st.markdown("""
    ### ✅ Model file found!
    
    **Next steps to make it work with the U-Net app:**
    1. **Note the filename** that worked: `{loaded_file}`
    2. **Rename it** to `unet_model.pth` or update the app to use your filename
    3. **Update the U-Net architecture** in the app to match your model
    """)

# File upload option for testing
st.header("📤 Alternative: Upload Model File")
uploaded_model = st.file_uploader(
    "Or upload your model file directly", 
    type=['pth', 'pt', 'pkl'],
    help="If your model file isn't in the repo, upload it here to test"
)

if uploaded_model is not None:
    try:
        # Save uploaded file temporarily
        with open("uploaded_model.pth", "wb") as f:
            f.write(uploaded_model.getvalue())
        
        # Try to load it
        checkpoint = torch.load("uploaded_model.pth", map_location='cpu')
        st.success("✅ Uploaded model loaded successfully!")
        st.write("Checkpoint type:", type(checkpoint))
        
        if isinstance(checkpoint, dict):
            st.write("Keys:", list(checkpoint.keys()))
            
    except Exception as e:
        st.error(f"❌ Failed to load uploaded model: {str(e)}")

# Create a simple test model if needed
st.header("🛠️ Troubleshooting Tools")

if st.button("Create a Test U-Net Model"):
    try:
        from model import UNet  # Assuming you have the UNet class defined
        
        # Create a simple U-Net model
        model = UNet(in_channels=3, out_channels=1)
        
        # Save it
        torch.save(model.state_dict(), 'test_unet_model.pth')
        st.success("✅ Created test model: `test_unet_model.pth`")
        st.info("Now try running the main U-Net app again!")
        
    except Exception as e:
        st.error(f"Failed to create test model: {str(e)}")
