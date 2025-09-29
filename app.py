import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Oil Spill Debug", layout="wide")

st.title("🔧 Oil Spill Detection Debug")
st.write("Let's diagnose why the model is detecting incorrectly")

@st.cache_resource
def load_model():
    try:
        model = torch.load('oil_spill_model_deploy.pth', map_location='cpu')
        if hasattr(model, 'eval'):
            model.eval()
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

model = load_model()

if model is not None:
    st.success("✅ Model loaded")
else:
    st.error("❌ Model failed to load")
    st.stop()

# Enhanced preprocessing with multiple options
def preprocess_image_debug(image, method='standard'):
    """Try different preprocessing methods"""
    original_size = image.size
    
    # Resize to common segmentation sizes
    if method == 'large':
        target_size = (512, 512)
    elif method == 'small':
        target_size = (224, 224)
    else:  # standard
        target_size = (256, 256)
    
    image_resized = image.resize(target_size)
    img_array = np.array(image_resized) / 255.0
    
    # Different normalization strategies
    if method == 'no_normalize':
        # No normalization
        img_tensor = torch.from_numpy(img_array).float()
    elif method == 'simple_normalize':
        # Simple 0-1 normalization
        img_tensor = torch.from_numpy(img_array).float()
    else:  # standard imagenet
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        img_tensor = torch.from_numpy(img_array).float()
    
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    return img_tensor, original_size, target_size

# Analysis functions
def analyze_prediction(output):
    """Analyze model output to understand what's happening"""
    if isinstance(output, torch.Tensor):
        output_np = output.squeeze().cpu().numpy()
    else:
        output_np = output
    
    analysis = {
        'min_value': float(np.min(output_np)),
        'max_value': float(np.max(output_np)),
        'mean_value': float(np.mean(output_np)),
        'std_value': float(np.std(output_np)),
        'median_value': float(np.median(output_np)),
        'percentile_95': float(np.percentile(output_np, 95)),
        'percentile_99': float(np.percentile(output_np, 99)),
        'above_0.5': float(np.sum(output_np > 0.5) / output_np.size * 100),
        'above_0.8': float(np.sum(output_np > 0.8) / output_np.size * 100),
    }
    return analysis, output_np

def create_confidence_histogram(prob_map):
    """Create histogram of confidence values"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(prob_map.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Confidence Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Prediction Confidences')
    ax.grid(True, alpha=0.3)
    return fig

# Main debug interface
st.header("🎯 Debug Prediction Issues")

uploaded_file = st.file_uploader(
    "Upload problematic image", 
    type=['png', 'jpg', 'jpeg', 'tiff', 'bmp']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Preprocessing options
    st.sidebar.header("🛠️ Preprocessing Options")
    preprocessing_method = st.sidebar.selectbox(
        "Preprocessing Method",
        ['standard', 'no_normalize', 'simple_normalize', 'large', 'small'],
        help="Try different preprocessing strategies"
    )
    
    # Multiple threshold options
    st.sidebar.header("🎚️ Threshold Options")
    threshold_method = st.sidebar.radio(
        "Threshold Method",
        ['fixed', 'adaptive', 'percentile']
    )
    
    if threshold_method == 'fixed':
        confidence_threshold = st.sidebar.slider("Fixed Threshold", 0.0, 1.0, 0.5, 0.01)
    elif threshold_method == 'adaptive':
        adaptive_factor = st.sidebar.slider("Adaptive Factor", 0.5, 3.0, 1.5, 0.1)
    else:  # percentile
        percentile = st.sidebar.slider("Percentile Threshold", 80, 99, 95)
    
    # Test button
    if st.button("🔍 Run Detailed Analysis"):
        with st.spinner("Analyzing model behavior..."):
            # Test different preprocessing
            input_tensor, original_size, target_size = preprocess_image_debug(image, preprocessing_method)
            
            # Get prediction
            with torch.no_grad():
                try:
                    output = model(input_tensor)
                    st.success("✅ Prediction completed")
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.stop()
            
            # Analyze output
            analysis, prob_map = analyze_prediction(output)
            
            # Display analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Prediction Statistics")
                for key, value in analysis.items():
                    st.write(f"**{key}:** {value:.4f}")
                
                # Interpretation
                st.subheader("💡 Interpretation")
                if analysis['max_value'] < 0.1:
                    st.warning("⚠️ Model is very uncertain (max confidence < 0.1)")
                elif analysis['max_value'] > 0.9 and analysis['above_0.8'] > 10:
                    st.warning("⚠️ Model is overconfident (many high confidence predictions)")
                elif analysis['mean_value'] > 0.7:
                    st.info("ℹ️ Model is generally confident")
                else:
                    st.info("ℹ️ Model shows moderate confidence")
            
            with col2:
                st.subheader("📈 Confidence Distribution")
                fig = create_confidence_histogram(prob_map)
                st.pyplot(fig)
            
            # Calculate adaptive threshold if needed
            if threshold_method == 'adaptive':
                confidence_threshold = analysis['mean_value'] * adaptive_factor
                confidence_threshold = min(0.99, max(0.01, confidence_threshold))
            elif threshold_method == 'percentile':
                confidence_threshold = analysis['percentile_95'] / 100.0 if percentile == 95 else analysis['percentile_99'] / 100.0
            
            st.info(f"**Using threshold: {confidence_threshold:.3f}**")
            
            # Create masks with different thresholds for comparison
            st.subheader("🔍 Compare Different Thresholds")
            
            thresholds_to_test = [
                confidence_threshold * 0.5,  # More sensitive
                confidence_threshold,        # Chosen threshold
                confidence_threshold * 1.5,  # More conservative
            ]
            
            cols = st.columns(3)
            for idx, (col, test_threshold) in enumerate(zip(cols, thresholds_to_test)):
                with col:
                    # Create mask
                    binary_mask = (prob_map > test_threshold).astype(np.uint8) * 255
                    binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
                    
                    # Create overlay
                    original_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    colored_mask = np.zeros_like(original_cv)
                    colored_mask[binary_mask_resized > 0] = [0, 0, 255]
                    blended = cv2.addWeighted(original_cv, 0.7, colored_mask, 0.3, 0)
                    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
                    
                    st.image(blended_rgb, use_column_width=True, caption=f"Threshold: {test_threshold:.3f}")
                    
                    # Stats for this threshold
                    oil_pixels = np.sum(binary_mask_resized > 0)
                    total_pixels = binary_mask_resized.size
                    coverage = (oil_pixels / total_pixels) * 100
                    st.write(f"Coverage: {coverage:.2f}%")
                    st.write(f"Pixels: {oil_pixels:,}")
            
            # Show raw probability map
            st.subheader("🎨 Raw Probability Visualization")
            
            col_prob1, col_prob2 = st.columns(2)
            
            with col_prob1:
                # Heatmap of probabilities
                prob_display = (prob_map * 255).astype(np.uint8)
                prob_colored = cv2.applyColorMap(prob_display, cv2.COLORMAP_JET)
                prob_resized = cv2.resize(prob_colored, original_size)
                st.image(prob_resized, use_column_width=True, caption="Probability Heatmap")
            
            with col_prob2:
                # Binary mask with chosen threshold
                binary_mask = (prob_map > confidence_threshold).astype(np.uint8) * 255
                binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
                st.image(binary_mask_resized, use_column_width=True, caption=f"Final Mask (threshold: {confidence_threshold:.3f})")
            
            # Model output insights
            st.subheader("🤖 Model Output Insights")
            
            if analysis['mean_value'] < 0.1:
                st.error("""
                **Issue:** Model is producing very low confidence values.
                **Possible causes:**
                - Wrong preprocessing (normalization)
                - Input size mismatch
                - Model expects different input format
                - Model was trained on very different data
                """)
            elif analysis['mean_value'] > 0.9:
                st.warning("""
                **Issue:** Model is overconfident.
                **Possible causes:**
                - Threshold too low
                - Model overfitting
                - Training/test distribution mismatch
                """)
            else:
                st.success("""
                **Model output looks reasonable.**
                Try adjusting the threshold to improve detection quality.
                """)

# Common solutions
st.header("🔧 Common Fixes for Wrong Detections")

with st.expander("1. Threshold Adjustment"):
    st.markdown("""
    **Problem:** Default threshold (0.5) might not be optimal.
    **Solution:** 
    - Use the debug tool above to find optimal threshold
    - Try values between 0.1 and 0.9
    - Use adaptive thresholding based on image statistics
    """)

with st.expander("2. Preprocessing Issues"):
    st.markdown("""
    **Problem:** Wrong image preprocessing.
    **Solutions:**
    - Try different normalization methods
    - Test different input sizes (224, 256, 512)
    - Ensure RGB format (not BGR)
    - Check if model expects specific normalization
    """)

with st.expander("3. Model Calibration"):
    st.markdown("""
    **Problem:** Model outputs are not well-calibrated.
    **Solutions:**
    - Use temperature scaling if available
    - Apply post-processing filters
    - Use morphological operations to clean masks
    """)

with st.expander("4. Input Quality"):
    st.markdown("""
    **Problem:** Input images differ from training data.
    **Solutions:**
    - Ensure similar lighting/conditions
    - Check image resolution and quality
    - Verify the expected input format
    """)
