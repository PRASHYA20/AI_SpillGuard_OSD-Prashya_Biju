import os
# Disable the problematic timm features
os.environ['TIMM_DISABLE_TEARDOWN'] = '1'
os.environ['TIMM_FUSED_ATTN'] = '0'

import streamlit as st
import torch
from PIL import Image
import numpy as np
import io
import requests
import os

# -----------------------------
# Dropbox Model URL
# -----------------------------
MODEL_PATH = "oil_spill_model_deploy.pth"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/stl47n6ixrzv59xs2jt4m/oil_spill_model_deploy.pth?rlkey=rojyk0fq73mk8tai8jc3exrev&dl=1"

st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊", layout="wide")
st.title("🌊 Oil Spill Detection App")
st.write("App is loading...")

# Try to import segmentation models with error handling
try:
    import segmentation_models_pytorch as smp
    st.success("✅ ML libraries loaded successfully!")
    
    # Continue with your app logic here
    # ... [rest of your model loading and processing code]
    
except Exception as e:
    st.error(f"❌ Failed to load ML libraries: {e}")
    st.info("This is a known compatibility issue. The app will still load for basic functionality.")
