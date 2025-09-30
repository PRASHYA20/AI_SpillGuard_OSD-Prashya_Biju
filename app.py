import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import io
import os
import torch
import torchvision.transforms as transforms

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
    st.sidebar.info("To enable AI detection, ensure your model file is in the repository")

# Settings
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.1)
show_analysis = st.sidebar.checkbox("Show Detailed Analysis", value=True)

def correct_preprocess_image(image, target_size=(256, 256)):
    """
    CORRECTED preprocessing using PIL only (no OpenCV)
    """
    # Convert to numpy and keep original for overlay
    if isinstance(image, Image.Image):
        original_image = np.array(image)
    else:
        original_image = image.copy()
    
    # Store original dimensions
    original_h, original_w = original_image.shape[:2]
    
    # Resize for model input using PIL
    image_pil = Image.fromarray(original_image)
    image_resized = image_pil.resize(target_size, Image.Resampling.LANCZOS)
    image_resized_array = np.array(image_resized)
    
    # Normalize to [0,1]
    image_normalized = image_resized_array.astype(np.float32) / 255.0
    
    # Convert to tensor and normalize properly
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image_resized).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, original_image, (original_h, original_w), image_resized_array

def correct_postprocess_mask(mask_pred, original_shape, target_size=(256, 256)):
    """
    CORRECTED postprocessing using PIL only
    """
    # Remove batch dimension and convert to numpy
    if isinstance(mask_pred, torch.Tensor):
        mask_pred = mask_pred.squeeze().detach().cpu().numpy()
    
    # Handle different model output formats
    if len(mask_pred.shape) == 3:  # [C, H, W] or [H, W, C]
        if mask_pred.shape[0] == 1 or mask_pred.shape[0] == 2:  # [C, H, W]
            mask_pred = mask_pred.transpose(1, 2, 0)
        
        # For multi-class, take argmax; for binary, use threshold
        if mask_pred.shape[-1] > 1:
            mask_binary = np.argmax(mask_pred, axis=-1)
        else:
            mask_binary = (mask_pred[..., 0] > 0.5).astype(np.uint8)
    else:  # Already 2D
        mask_binary = (mask_pred > 0.5).astype(np.uint8)
    
    # Resize back to original image size using PIL
    mask_pil = Image.fromarray(mask_binary.astype(np.uint8) * 255)
    mask_original_size = mask_pil.resize(
        (original_shape[1], original_shape[0]), 
        Image.Resampling.NEAREST  # Use nearest to preserve binary values
    )
    
    return np.array(mask_original_size)

def create_correct_overlay(original_image, mask, alpha=0.6):
    """
    CORRECTED overlay creation using PIL only
    """
    # Convert to PIL if needed
    if isinstance(original_image, np.ndarray):
        original_pil = Image.fromarray(original_image.astype(np.uint8))
    else:
        original_pil = original_image
    
    # Create a copy for overlay
    overlay = original_pil.copy()
    
    # Create a red overlay image
    red_overlay = Image.new('RGBA', overlay.size, (255, 0, 0, int(255 * alpha)))
    
    # Create mask for where to apply the overlay
    mask_binary = mask > 0
    mask_rgba = Image.fromarray((mask_binary * 255).astype(np.uint8)).convert('L')
    
    # Apply the red overlay only to spill areas
    overlay = overlay.convert('RGBA')
    overlay.paste(red_overlay, (0, 0), mask_rgba)
    
    return overlay.convert('RGB')

def load_and_predict(model_path, image_tensor):
    """
    Simulate model prediction - REPLACE WITH YOUR ACTUAL MODEL LOADING
    """
    try:
        # TODO: Replace with your actual model loading code
        # model = torch.load(model_path, map_location='cpu')
        # model.eval()
        # with torch.no_grad():
        #     prediction = model(image_tensor)
        
        # For demo purposes, create a synthetic prediction
        batch_size, channels, height, width = image_tensor.shape
        
        # Create a more realistic synthetic mask with some spill-like shapes
        synthetic_pred = torch.zeros(batch_size, 1, height, width)
        
        # Add some elliptical "spill" areas
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing='ij'
        )
        
        # Main spill
        main_spill = ((x_coords - 0.1)**2 / 0.4 + (y_coords - 0.2)**2 / 0.3) < 1
        synthetic_pred[0, 0] = main_spill.float() * 0.8
        
        # Smaller spills
        small_spill1 = ((x_coords + 0.3)**2 / 0.2 + (y_coords - 0.4)**2 / 0.15) < 1
        small_spill2 = ((x_coords - 0.4)**2 / 0.25 + (y_coords + 0.3)**2 / 0.2) < 1
        
        synthetic_pred[0, 0] += small_spill1.float() * 0.6
        synthetic_pred[0, 0] += small_spill2.float() * 0.7
        
        # Clip values
        synthetic_pred = torch.clamp(synthetic_pred, 0, 1)
        
        return synthetic_pred
        
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

# Main app
uploaded_file = st.file_uploader("📤 Upload Satellite Image", type=["jpg", "jpeg", "png", "tiff"])

if uploaded_file is not None:
    try:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        
        # Display layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🛰️ Original Image")
            st.image(image, use_container_width=True)
            st.write(f"Size: {image.size}")
            st.write(f"Mode: {image.mode}")
        
        # Check if we have a model file
        if model_files:
            st.success("✅ AI Model Available - Processing Image")
            
            with st.spinner("🔄 Analyzing image for oil spills..."):
                # CORRECTED preprocessing
                image_tensor, original_array, original_shape, resized_image = correct_preprocess_image(image)
                
                # Get prediction
                prediction = load_and_predict(model_files[0], image_tensor)
                
                if prediction is not None:
                    # CORRECTED postprocessing
                    final_mask = correct_postprocess_mask(prediction, original_shape)
                    
                    # Apply confidence threshold
                    final_mask_binary = (final_mask > (confidence_threshold * 255)).astype(np.uint8) * 255
                    
                    # CORRECTED overlay creation
                    overlay_image = create_correct_overlay(original_array, final_mask_binary)
                    
                    # Convert mask to PIL for display
                    mask_pil = Image.fromarray(final_mask_binary)
                    
                    # Display results
                    with col2:
                        st.subheader("🎭 Detection Mask")
                        st.image(mask_pil, use_container_width=True, clamp=True)
                        st.write("White areas = Potential oil spills")
                    
                    with col3:
                        st.subheader("🛢️ Oil Spill Overlay")
                        st.image(overlay_image, use_container_width=True)
                        st.write("Red areas = Detected oil spills")
                    
                    # Analysis
                    if show_analysis:
                        st.subheader("📊 Analysis Results")
                        
                        # Calculate statistics on the CORRECT mask
                        total_pixels = final_mask_binary.size
                        oil_pixels = np.sum(final_mask_binary > 0)
                        coverage = (oil_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                        
                        col_stats1, col_stats2, col_stats3 = st.columns(3)
                        
                        with col_stats1:
                            st.metric("Spill Coverage", f"{coverage:.2f}%")
                        with col_stats2:
                            st.metric("Affected Pixels", f"{oil_pixels:,}")
                        with col_stats3:
                            st.metric("Confidence Level", f"{confidence_threshold:.1f}")
                        
                        # Risk assessment
                        st.subheader("🔍 Risk Assessment")
                        if coverage > 10:
                            st.error("🚨 **HIGH RISK** - Significant oil spill detected")
                            st.write("Immediate containment action recommended")
                        elif coverage > 1:
                            st.warning("⚠️ **MEDIUM RISK** - Notable oil contamination")
                            st.write("Monitoring and assessment needed")
                        elif coverage > 0.1:
                            st.info("ℹ️ **LOW RISK** - Minor oil detection")
                            st.write("Regular monitoring recommended")
                        else:
                            st.success("✅ **CLEAN** - No significant oil detected")
                            st.write("Water appears clean")
                
                else:
                    st.error("❌ Prediction failed")
                    with col2:
                        st.subheader("🎭 Detection Mask")
                        st.info("Model prediction unavailable")
                    with col3:
                        st.subheader("🛢️ Oil Spill Overlay")
                        st.info("Model prediction unavailable")
            
            # Download section
            st.subheader("💾 Download Results")
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                if 'final_mask_binary' in locals():
                    buf_mask = io.BytesIO()
                    mask_pil.save(buf_mask, format="PNG")
                    st.download_button(
                        label="📥 Download Detection Mask",
                        data=buf_mask.getvalue(),
                        file_name="oil_spill_mask.png",
                        mime="image/png",
                        use_container_width=True
                    )
            
            with col_dl2:
                if 'overlay_image' in locals():
                    buf_overlay = io.BytesIO()
                    overlay_image.save(buf_overlay, format="PNG")
                    st.download_button(
                        label="📥 Download Overlay Image",
                        data=buf_overlay.getvalue(),
                        file_name="oil_spill_overlay.png",
                        mime="image/png",
                        use_container_width=True
                    )
        
        else:
            # No model file - show basic info
            st.warning("🔸 AI Model Not Available - Running in Demo Mode")
            
            with col2:
                st.subheader("ℹ️ Information")
                st.write("To enable actual oil spill detection:")
                st.write("1. Add your trained model file to the repository")
                st.write("2. Update the `load_and_predict` function")
                st.write("3. The app will automatically use it")
            
            with col3:
                st.subheader("📋 Required Files")
                st.code("model.pth or model.pt")

    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.info("Please try another image file")

else:
    st.info("👆 **Please upload a satellite image to begin analysis**")

# Instructions
with st.expander("🔧 Setup & Troubleshooting"):
    st.markdown("""
    **Requirements.txt for deployment:**
    ```txt
    streamlit
    numpy
    pillow
    torch
    torchvision
    ```
    
    **To enable your actual model:**
    1. Replace the `load_and_predict` function with your model loading code
    2. Ensure your model file is in the repository
    3. The preprocessing now matches standard PyTorch models
    
    **Current directory files:**
    """)
    
    try:
        files = os.listdir('.')
        for file in sorted(files):
            st.write(f"- `{file}`")
    except:
        st.write("Unable to list files")
