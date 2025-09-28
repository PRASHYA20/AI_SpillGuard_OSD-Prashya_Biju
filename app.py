# app.py
from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename
import io
from PIL import Image
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tiff'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained model (you'll need to train or obtain this model)
# For demonstration, we'll create a placeholder function
def load_oil_spill_model():
    """
    Load the oil spill detection model.
    In practice, you would load a pre-trained TensorFlow/Keras model.
    """
    try:
        # Replace with your actual model path
        # model = load_model('oil_spill_model.h5')
        # return model
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Initialize model
model = load_oil_spill_model()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    """Preprocess the uploaded image for prediction"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")
    
    # Resize to expected input size (adjust based on your model)
    img = cv2.resize(img, (256, 256))
    
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def detect_oil_spill_areas(image_path):
    """
    Detect oil spill areas in the image.
    This is a simplified version - in practice, you'd use your trained model.
    """
    # Read and preprocess image
    img = cv2.imread(image_path)
    original_shape = img.shape
    
    # Resize for processing
    img_resized = cv2.resize(img, (256, 256))
    
    # Convert to HSV color space for better color-based segmentation
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for oil spill detection (these are approximate)
    # Oil spills often appear as dark, smooth areas
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 100])
    
    # Create mask for dark areas
    mask = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    min_area = 100  # Minimum area threshold
    oil_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Create result image with detected areas
    result_img = img_resized.copy()
    cv2.drawContours(result_img, oil_contours, -1, (0, 0, 255), 2)  # Red contours
    
    # Calculate total oil spill area percentage
    total_pixels = mask.shape[0] * mask.shape[1]
    oil_pixels = np.sum(mask > 0)
    oil_percentage = (oil_pixels / total_pixels) * 100
    
    return result_img, oil_percentage, len(oil_contours)

def create_visualization(original_img, result_img, oil_percentage, contour_count):
    """Create a side-by-side visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Result image
    ax2.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Detected Oil Spills\n({oil_percentage:.2f}% area, {contour_count} regions)')
    ax2.axis('off')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return buf

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Detect oil spill areas
            original_img = cv2.imread(filepath)
            original_img_resized = cv2.resize(original_img, (256, 256))
            result_img, oil_percentage, contour_count = detect_oil_spill_areas(filepath)
            
            # Create visualization
            viz_buffer = create_visualization(original_img_resized, result_img, oil_percentage, contour_count)
            
            # Convert visualization to base64 for display
            viz_base64 = base64.b64encode(viz_buffer.getvalue()).decode('utf-8')
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'oil_percentage': round(oil_percentage, 2),
                'contour_count': contour_count,
                'visualization': f'data:image/png;base64,{viz_base64}',
                'message': f'Detected {oil_percentage:.2f}% potential oil spill area across {contour_count} regions'
            })
            
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle multiple file uploads"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Save uploaded file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Detect oil spill areas
                _, oil_percentage, contour_count = detect_oil_spill_areas(filepath)
                
                results.append({
                    'filename': filename,
                    'oil_percentage': round(oil_percentage, 2),
                    'contour_count': contour_count,
                    'status': 'High risk' if oil_percentage > 5 else 'Low risk'
                })
                
                # Clean up
                os.remove(filepath)
                
            except Exception as e:
                results.append({
                    'filename': filename,
                    'error': str(e),
                    'status': 'Error'
                })
    
    return jsonify({'results': results})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    return predict()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
