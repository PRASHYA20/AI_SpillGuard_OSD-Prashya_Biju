import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import requests
import torch
import torch.nn as nn
from torchvision import models
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="AI Oil Spill Detection",
    page_icon="🛢️",
    layout="wide"
)

class OilSpillDetector:
    def __init__(self):
        self.model = None
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        st.title("🛢️ AI Oil Spill Detection System")
        st.markdown("---")
        
        # Sidebar
        with st.sidebar:
            st.header("Controls")
            
            uploaded_file = st.file_uploader(
                "Upload Image", 
                type=['jpg', 'jpeg', 'png'],
                help="Upload an image for oil spill detection"
            )
            
            st.subheader("Detection Settings")
            
            # Confidence threshold
            confidence = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.3,
                step=0.1,
                help="Adjust detection sensitivity"
            )
            
            # Minimum area
            min_area = st.slider(
                "Minimum Area",
                min_value=50,
                max_value=2000,
                value=100,
                step=50,
                help="Minimum oil spill area to detect"
            )
            
            # Process button
            process_btn = st.button("Detect Oil Spills", type="primary")
            
            st.markdown("---")
            
            if self.model is not None:
                st.success("✅ AI Model Loaded")
            else:
                st.warning("⚠️ Using Basic Detection")
        
        # Main content
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            if uploaded_file is not None:
                original_image = Image.open(uploaded_file)
                st.image(original_image, use_column_width=True)
                original_cv = self.pil_to_cv2(original_image)
            else:
                original_cv = None
                st.info("Please upload an image")
        
        with col2:
            st.subheader("Detection Results")
            if uploaded_file is not None and process_btn:
                with st.spinner("Detecting oil spills..."):
                    try:
                        # Try AI model first
                        if self.model is not None:
                            result_image, stats = self.detect_with_model(original_cv, confidence, min_area)
                            method = "AI Model"
                        else:
                            result_image, stats = self.basic_detection(original_cv, min_area)
                            method = "Basic CV"
                        
                        st.image(result_image, use_column_width=True)
                        self.show_stats(stats, method)
                        self.download_result(result_image)
                        
                    except Exception as e:
                        st.error(f"Detection error: {str(e)}")
                        # Fallback to basic detection
                        result_image, stats = self.basic_detection(original_cv, min_area)
                        st.image(result_image, use_column_width=True)
                        self.show_stats(stats, "Basic CV (Fallback)")
                        self.download_result(result_image)
                        
            elif uploaded_file is not None:
                st.info("Click 'Detect Oil Spills' to analyze")
            else:
                st.info("Results will appear here")
    
    def load_model(self):
        """Load model with better error handling"""
        try:
            model_url = "https://www.dropbox.com/scl/fi/stl47n6ixrzv59xs2jt4m/oil_spill_model_deploy.pth?rlkey=rojyk0fq73mk8tai8jc3exrev&st=g0afxvrw&dl=1"
            
            with st.sidebar.spinner("Downloading model..."):
                model_path = self.download_model(model_url)
                
            with st.sidebar.spinner("Loading model..."):
                self.model = self.load_pytorch_model(model_path)
                
        except Exception as e:
            st.sidebar.error(f"Model load failed: {str(e)}")
            self.model = None
    
    def download_model(self, url):
        """Download model from Dropbox"""
        temp_dir = tempfile.gettempdir()
        model_path = os.path.join(temp_dir, "oil_model.pth")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        return model_path
    
    def load_pytorch_model(self, model_path):
        """Load PyTorch model with multiple fallback methods"""
        device = torch.device('cpu')  # Use CPU for compatibility
        
        try:
            # Try loading as full model first
            model = torch.load(model_path, map_location=device)
            
            # If it's a state dict, create model architecture first
            if isinstance(model, dict):
                # Try common segmentation architectures
                try:
                    model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=1)
                    if 'state_dict' in model:
                        model.load_state_dict(model['state_dict'])
                    else:
                        model.load_state_dict(model)
                except:
                    model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)
                    model.load_state_dict(model)
            
            model.eval()
            return model
            
        except Exception as e:
            st.error(f"Model loading detailed error: {str(e)}")
            return None
    
    def detect_with_model(self, image, confidence=0.3, min_area=100):
        """Detect using AI model with proper preprocessing"""
        try:
            # Preprocess for model
            input_tensor = self.preprocess_for_inference(image)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                
                # Handle different output types
                if isinstance(output, dict):
                    prediction = output['out']
                else:
                    prediction = output
                
                # Get probability mask
                if prediction.shape[1] == 1:
                    prob_mask = torch.sigmoid(prediction)[0, 0]
                else:
                    prob_mask = torch.softmax(prediction, dim=1)[0, 1]
                
                mask_np = prob_mask.cpu().numpy()
            
            # Resize to original
            h, w = image.shape[:2]
            mask_resized = cv2.resize(mask_np, (w, h))
            
            # Apply threshold
            binary_mask = (mask_resized > confidence).astype(np.uint8) * 255
            
            # Clean up mask
            clean_mask = self.clean_mask(binary_mask, min_area)
            
            # Create result
            result = self.create_result_image(image, clean_mask)
            stats = self.calculate_stats(clean_mask)
            
            return result, stats
            
        except Exception as e:
            st.error(f"Model inference error: {str(e)}")
            raise
    
    def preprocess_for_inference(self, image):
        """Proper preprocessing for model inference"""
        # Resize to common segmentation size
        resized = cv2.resize(image, (512, 512))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # Convert to tensor
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
        
        return tensor
    
    def basic_detection(self, image, min_area=100):
        """Fallback basic detection using color and texture analysis"""
        # Convert color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Oil color ranges (adjust based on your images)
        # Dark, low saturation areas often indicate oil
        hsv_mask1 = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 80, 120]))
        hsv_mask2 = cv2.inRange(hsv, np.array([100, 0, 0]), np.array([140, 80, 100]))
        
        # LAB space for dark regions
        lab_mask = cv2.inRange(lab, np.array([0, 120, 120]), np.array([255, 140, 140]))
        
        # Combine masks
        combined = cv2.bitwise_or(hsv_mask1, hsv_mask2)
        combined = cv2.bitwise_or(combined, lab_mask)
        
        # Clean up
        kernel = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Filter by area
        final_mask = self.clean_mask(cleaned, min_area)
        
        # Create result
        result = self.create_result_image(image, final_mask)
        stats = self.calculate_stats(final_mask)
        
        return result, stats
    
    def clean_mask(self, mask, min_area):
        """Clean up mask using contour analysis"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.drawContours(clean_mask, [contour], -1, 255, -1)
        
        return clean_mask
    
    def create_result_image(self, original, mask):
        """Create result image with black oil spill areas"""
        result = original.copy()
        result[mask == 255] = 0  # Make oil areas black
        return result
    
    def calculate_stats(self, mask):
        """Calculate detection statistics"""
        total_pixels = mask.shape[0] * mask.shape[1]
        oil_pixels = np.sum(mask == 255)
        oil_percent = (oil_pixels / total_pixels) * 100
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return {
            'total_pixels': total_pixels,
            'oil_pixels': oil_pixels,
            'oil_percent': oil_percent,
            'num_spills': len(contours),
            'contours': contours
        }
    
    def show_stats(self, stats, method):
        """Display statistics"""
        st.subheader("Detection Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Oil Coverage", f"{stats['oil_percent']:.2f}%")
        
        with col2:
            st.metric("Oil Area", f"{stats['oil_pixels']:,} px")
        
        with col3:
            st.metric("Number of Spills", stats['num_spills'])
        
        st.info(f"Detection Method: {method}")
        
        # Show contour details
        with st.expander("Contour Details"):
            for i, contour in enumerate(stats['contours']):
                area = cv2.contourArea(contour)
                st.write(f"Spill {i+1}: {area:.0f} pixels")
    
    def download_result(self, image):
        """Download result image"""
        pil_img = self.cv2_to_pil(image)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        
        st.download_button(
            "Download Result",
            data=buf,
            file_name="oil_detection.png",
            mime="image/png"
        )
    
    def pil_to_cv2(self, pil_img):
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def cv2_to_pil(self, cv_img):
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

# Run the app
if __name__ == "__main__":
    detector = OilSpillDetector()
