import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import base64
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import tempfile
import os
import urllib.parse

# Page configuration
st.set_page_config(
    page_title="AI Oil Spill Detection",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class OilSpillModel(nn.Module):
    def __init__(self, num_classes=1):
        super(OilSpillModel, self).__init__()
        # Using a pre-trained backbone for segmentation
        self.backbone = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
        
    def forward(self, x):
        return self.backbone(x)['out']

class OilSpillDetector:
    def __init__(self):
        self.model = None
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        # Main title
        st.title("🛢️ AI Oil Spill Detection System")
        st.markdown("---")
        
        # Sidebar for controls
        with st.sidebar:
            st.header("Controls")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload Image", 
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                help="Upload an image for oil spill detection"
            )
            
            st.subheader("Detection Method")
            
            # Detection method
            detection_method = st.selectbox(
                "Select Method",
                ["ai_model", "combined", "color_based", "edge_aware", "kmeans"],
                help="Choose the detection algorithm. AI Model uses your trained neural network."
            )
            
            if detection_method == "ai_model":
                # Confidence threshold for AI model
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.5,
                    step=0.1,
                    help="Minimum confidence for AI model detection"
                )
            else:
                # Sensitivity (for traditional methods)
                sensitivity = st.slider(
                    "Detection Sensitivity",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="Higher values detect more potential oil areas"
                )
            
            # Minimum area
            min_area = st.slider(
                "Minimum Area Threshold",
                min_value=100,
                max_value=5000,
                value=500,
                step=100,
                help="Filter out small detections below this pixel area"
            )
            
            # Visualization style
            visualization_style = st.selectbox(
                "Visualization Style",
                ["black_areas", "red_overlay", "binary_mask"],
                help="How to visualize detected oil spills"
            )
            
            # Process button
            process_btn = st.button(
                "Detect Oil Spills",
                type="primary",
                use_container_width=True
            )
            
            st.markdown("---")
            
            # Model status
            if self.model is not None:
                st.success("✅ AI Model Loaded Successfully")
            else:
                st.warning("⚠️ AI Model Not Loaded - Using Traditional Methods")
            
            st.subheader("About")
            st.info(
                "This app uses AI and computer vision to detect oil spills in images. "
                "Oil spills are shown in black areas for clear visualization."
            )
        
        # Main content area
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            if uploaded_file is not None:
                # Display original image
                original_image = Image.open(uploaded_file)
                st.image(original_image, caption="Uploaded Image", use_column_width=True)
                
                # Convert to OpenCV format
                original_cv = self.pil_to_cv2(original_image)
            else:
                original_cv = None
                st.info("Please upload an image to begin analysis")
        
        with col2:
            st.subheader("Detection Results")
            if uploaded_file is not None and process_btn:
                with st.spinner("Analyzing image for oil spills..."):
                    try:
                        if detection_method == "ai_model" and self.model is not None:
                            # Use AI model
                            confidence_threshold = confidence_threshold if 'confidence_threshold' in locals() else 0.5
                            result_image, mask, stats = self.detect_with_ai_model(
                                original_cv, confidence_threshold, min_area
                            )
                            method_used = "AI Model"
                        else:
                            # Use traditional methods
                            sensitivity = sensitivity if 'sensitivity' in locals() else 0.7
                            result_image, mask, stats = self.detect_oil_spills(
                                original_cv, detection_method, sensitivity, min_area
                            )
                            method_used = detection_method.replace("_", " ").title()
                        
                        # Apply visualization style
                        result_image = self.apply_visualization_style(
                            original_cv, mask, visualization_style
                        )
                        
                        # Display results
                        st.image(result_image, caption=f"Oil Spill Detection ({method_used}) - Oil shown in black", use_column_width=True)
                        
                        # Show statistics
                        self.display_statistics(stats, method_used)
                        
                        # Download button
                        self.create_download_button(result_image)
                        
                    except Exception as e:
                        st.error(f"Error during detection: {str(e)}")
                        st.exception(e)
            elif uploaded_file is not None:
                st.info("Click 'Detect Oil Spills' to analyze the image")
            else:
                st.info("Results will appear here after processing")
    
    def apply_visualization_style(self, original_image, mask, style):
        """Apply different visualization styles for oil spill detection"""
        if style == "black_areas":
            # Show oil spills as black areas on original image
            result_image = original_image.copy()
            result_image[mask == 255] = [0, 0, 0]  # Black for oil spills
            return result_image
            
        elif style == "red_overlay":
            # Show oil spills with red transparent overlay
            result_image = original_image.copy()
            overlay = result_image.copy()
            overlay[mask == 255] = [0, 0, 255]  # Red for oil
            alpha = 0.3
            result_image = cv2.addWeighted(overlay, alpha, original_image, 1 - alpha, 0)
            return result_image
            
        elif style == "binary_mask":
            # Show pure binary mask (white oil spills on black background)
            result_image = np.zeros_like(original_image)
            result_image[mask == 255] = [255, 255, 255]  # White oil spills on black background
            return result_image
    
    def load_model(self):
        """Load your AI model from Dropbox"""
        try:
            # Your Dropbox model URL
            model_url = "https://www.dropbox.com/scl/fi/stl47n6ixrzv59xs2jt4m/oil_spill_model_deploy.pth?rlkey=rojyk0fq73mk8tai8jc3exrev&st=g0afxvrw&dl=1"
            
            with st.sidebar.spinner("Loading AI model from Dropbox..."):
                # Download model file
                model_path = self.download_file_from_dropbox(model_url, "oil_spill_model.pth")
                
                # Load the model
                self.model = self.load_pytorch_model(model_path)
                
                if self.model is not None:
                    st.sidebar.success("✅ AI Model Loaded Successfully")
                
        except Exception as e:
            st.sidebar.error(f"Failed to load AI model: {str(e)}")
            self.model = None
    
    def download_file_from_dropbox(self, url, filename):
        """Download file from Dropbox URL"""
        try:
            # Create temp directory
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, filename)
            
            # Use requests to download the file
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            
            # Check file size
            total_size = int(response.headers.get('content-length', 0))
            
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            with open(file_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(progress)
                            status_text.text(f"Downloading: {downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB")
            
            progress_bar.empty()
            status_text.empty()
            
            # Verify file was downloaded
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                st.sidebar.success(f"Model downloaded: {os.path.getsize(file_path)/(1024*1024):.1f}MB")
                return file_path
            else:
                raise Exception("Downloaded file is empty or doesn't exist")
                
        except Exception as e:
            st.sidebar.error(f"Download failed: {str(e)}")
            raise
    
    def load_pytorch_model(self, model_path):
        """Load PyTorch model with error handling for different formats"""
        try:
            # Try different loading methods
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Method 1: Try loading as a state dict
            try:
                checkpoint = torch.load(model_path, map_location=device)
                
                if isinstance(checkpoint, dict):
                    # It's likely a checkpoint dictionary
                    if 'model_state_dict' in checkpoint:
                        model = OilSpillModel(num_classes=1)
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model = OilSpillModel(num_classes=1)
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        # Try to load directly as state dict
                        model = OilSpillModel(num_classes=1)
                        model.load_state_dict(checkpoint)
                else:
                    # It's a full model
                    model = checkpoint
                
            except:
                # Method 2: Try loading with different parameters
                try:
                    model = torch.load(model_path, map_location=device)
                except:
                    # Method 3: Try loading as a DeepLabV3 model
                    model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=1)
                    checkpoint = torch.load(model_path, map_location=device)
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
            
            model.eval()
            model.to(device)
            
            st.sidebar.info(f"Model loaded on: {device}")
            return model
            
        except Exception as e:
            st.sidebar.error(f"Error loading PyTorch model: {str(e)}")
            return None
    
    def detect_with_ai_model(self, image, confidence_threshold=0.5, min_area=500):
        """Detect oil spills using AI model"""
        if self.model is None:
            raise ValueError("AI model not loaded")
        
        # Preprocess image for model
        input_tensor = self.preprocess_for_model(image)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Handle different output formats
            if isinstance(output, dict):
                prediction = output['out'] if 'out' in output else output
            else:
                prediction = output
            
            # Convert to probability mask
            if prediction.shape[1] == 1:  # Single channel (sigmoid)
                prob_mask = torch.sigmoid(prediction)[0, 0]
            else:  # Multiple channels (softmax)
                prob_mask = torch.softmax(prediction, dim=1)[0, 1]  # Assume class 1 is oil
            
            mask_np = prob_mask.cpu().numpy()
        
        # Resize back to original dimensions
        original_height, original_width = image.shape[:2]
        mask_resized = cv2.resize(mask_np, (original_width, original_height))
        
        # Apply confidence threshold
        binary_mask = (mask_resized > confidence_threshold).astype(np.uint8) * 255
        
        # Post-process mask
        refined_mask = self.refine_detection_mask(binary_mask, min_area)
        
        # Calculate statistics
        stats = self.calculate_statistics(refined_mask)
        
        return image, refined_mask, stats
    
    def preprocess_for_model(self, image):
        """Preprocess image for AI model"""
        # Resize to expected input size (typical for segmentation models)
        input_size = (512, 512)
        resized = cv2.resize(image, input_size)
        
        # Normalize (ImageNet stats are common for pre-trained models)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Convert to tensor and rearrange dimensions
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
        
        return tensor

    # Keep all the traditional detection methods
    def pil_to_cv2(self, pil_image):
        """Convert PIL Image to OpenCV format"""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def cv2_to_pil(self, cv_image):
        """Convert OpenCV Image to PIL format"""
        return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    
    def preprocess_image(self, image):
        """Enhanced preprocessing for better oil spill detection"""
        # Convert to appropriate color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Noise reduction
        denoised = cv2.medianBlur(enhanced, 5)
        
        return denoised, hsv, lab
    
    def color_based_detection(self, image, hsv, lab):
        """Color-based oil spill detection"""
        # HSV ranges for oil sheen
        hsv_lower1 = np.array([0, 0, 50])
        hsv_upper1 = np.array([180, 80, 200])
        
        hsv_lower2 = np.array([100, 0, 30])
        hsv_upper2 = np.array([140, 80, 150])
        
        # LAB ranges for oil
        lab_lower = np.array([0, 120, 120])
        lab_upper = np.array([255, 145, 145])
        
        # Create masks
        hsv_mask1 = cv2.inRange(hsv, hsv_lower1, hsv_upper1)
        hsv_mask2 = cv2.inRange(hsv, hsv_lower2, hsv_upper2)
        lab_mask = cv2.inRange(lab, lab_lower, lab_upper)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(hsv_mask1, hsv_mask2)
        combined_mask = cv2.bitwise_or(combined_mask, lab_mask)
        
        return combined_mask
    
    def edge_aware_detection(self, image, gray):
        """Edge-aware oil spill detection"""
        # Calculate gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gradient_magnitude, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return adaptive_thresh
    
    def kmeans_segmentation(self, image, k=4):
        """K-means clustering for oil spill segmentation"""
        # Reshape image to 2D array of pixels
        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        
        # Perform K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to 8-bit values
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(image.shape)
        
        return segmented_image, labels.reshape(image.shape[:2])
    
    def refine_detection_mask(self, mask, min_area=500):
        """Refine the detection mask using morphological operations and contour analysis"""
        # Morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and shape
        refined_mask = np.zeros_like(cleaned_mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > min_area:
                # Calculate contour properties
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                
                if hull_area > 0:
                    solidity = area / hull_area
                    
                    # Only keep contours with reasonable solidity
                    if solidity > 0.2:  # Lower threshold for irregular oil shapes
                        cv2.drawContours(refined_mask, [contour], -1, 255, -1)
        
        # Final smoothing
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
        
        return refined_mask
    
    def detect_oil_spills_combined(self, image, sensitivity=0.7):
        """Combined detection method using multiple approaches"""
        # Preprocessing
        processed, hsv, lab = self.preprocess_image(image)
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Get masks from different methods
        color_mask = self.color_based_detection(processed, hsv, lab)
        edge_mask = self.edge_aware_detection(processed, gray)
        
        # K-means segmentation
        segmented, labels = self.kmeans_segmentation(processed, k=4)
        
        # Assume oil is in the darker clusters
        kmeans_mask = np.zeros_like(gray)
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            cluster_mean = np.mean(gray[labels == label])
            if cluster_mean < 100:  # Dark clusters might be oil
                kmeans_mask[labels == label] = 255
        
        # Combine masks with weights
        combined_mask = color_mask.astype(float) * 0.5 + \
                       edge_mask.astype(float) * 0.3 + \
                       kmeans_mask.astype(float) * 0.2
        
        # Normalize and threshold
        combined_mask = cv2.normalize(combined_mask, None, 0, 255, cv2.NORM_MINMAX)
        _, final_mask = cv2.threshold(combined_mask, sensitivity * 255, 255, cv2.THRESH_BINARY)
        
        final_mask = final_mask.astype(np.uint8)
        
        return final_mask
    
    def detect_oil_spills(self, image, method="combined", sensitivity=0.7, min_area=500):
        """Traditional detection methods"""
        if method == "color_based":
            processed, hsv, lab = self.preprocess_image(image)
            mask = self.color_based_detection(processed, hsv, lab)
        elif method == "edge_aware":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = self.edge_aware_detection(image, gray)
        elif method == "kmeans":
            segmented, labels = self.kmeans_segmentation(image)
            mask = np.zeros_like(image[:,:,0])
            # Select darker clusters as potential oil
            for label in np.unique(labels):
                if np.mean(image[labels == label]) < 100:
                    mask[labels == label] = 255
        else:  # combined
            mask = self.detect_oil_spills_combined(image, sensitivity)
        
        # Refine the detection
        refined_mask = self.refine_detection_mask(mask, min_area)
        
        # Calculate statistics
        stats = self.calculate_statistics(refined_mask)
        
        return image, refined_mask, stats
    
    def calculate_statistics(self, mask):
        """Calculate detection statistics"""
        total_pixels = mask.shape[0] * mask.shape[1]
        oil_pixels = np.sum(mask == 255)
        oil_percentage = (oil_pixels / total_pixels) * 100
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stats = {
            'total_pixels': total_pixels,
            'oil_pixels': oil_pixels,
            'oil_percentage': oil_percentage,
            'num_spills': len(contours),
            'contours': contours
        }
        
        return stats
    
    def display_statistics(self, stats, method_used):
        """Display detection statistics"""
        st.subheader("Detection Statistics")
        
        st.info(f"**Method Used:** {method_used}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Oil Coverage",
                f"{stats['oil_percentage']:.2f}%"
            )
        
        with col2:
            st.metric(
                "Contaminated Area",
                f"{stats['oil_pixels']:,} px"
            )
        
        with col3:
            st.metric(
                "Number of Spills",
                f"{stats['num_spills']}"
            )
        
        # Detailed analysis expander
        with st.expander("Detailed Analysis"):
            if stats['contours']:
                st.write("**Individual Spill Analysis:**")
                for i, contour in enumerate(stats['contours']):
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                    else:
                        circularity = 0
                    
                    st.write(
                        f"Spill {i+1}: Area={area:.0f} px, "
                        f"Perimeter={perimeter:.0f} px, "
                        f"Circularity={circularity:.3f}"
                    )
            else:
                st.write("No significant oil spills detected.")
    
    def create_download_button(self, image):
        """Create download button for processed image"""
        # Convert to PIL
        pil_image = self.cv2_to_pil(image)
        
        # Convert to bytes
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)
        
        # Create download button
        st.download_button(
            label="Download Result",
            data=buf,
            file_name="oil_spill_detection.png",
            mime="image/png",
            use_container_width=True
        )

# Run the app
if __name__ == "__main__":
    detector = OilSpillDetector()
