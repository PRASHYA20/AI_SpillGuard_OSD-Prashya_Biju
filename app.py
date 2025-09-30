import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import io
import collections
import os

# Set page config
st.set_page_config(page_title="Oil Spill Shape Detection", layout="wide")

st.title("🎯 Oil Spill Shape Detection")
st.write("Extract precise oil spill shapes and contours")

# First, let's debug the file situation
st.sidebar.header("🔍 File Status")

def check_model_files():
    """Check what model files exist"""
    all_files = os.listdir('.')
    model_files = [f for f in all_files if f.endswith(('.pth', '.pt', '.pkl'))]
    
    if model_files:
        st.sidebar.success(f"✅ Found {len(model_files)} model file(s)")
        for file in model_files:
            size_mb = os.path.getsize(file) / (1024 * 1024)
            st.sidebar.write(f"📦 {file} ({size_mb:.1f} MB)")
        return model_files
    else:
        st.sidebar.error("❌ No model files found!")
        st.sidebar.info("""
        **To fix:**
        1. Ensure your model file is in the same directory
        2. File should be: .pth, .pt, or .pkl
        3. Use the upload section below
        """)
        return []

# Check files
model_files = check_model_files()

# Model architecture
class OilSpillModel(nn.Module):
    def __init__(self, num_classes=1):
        super(OilSpillModel, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return torch.sigmoid(x)

@st.cache_resource
def load_model():
    """Try to load the model file"""
    if not model_files:
        return None
    
    for model_file in model_files:
        try:
            st.sidebar.info(f"🔄 Loading: {model_file}")
            checkpoint = torch.load(model_file, map_location='cpu')
            
            if isinstance(checkpoint, collections.OrderedDict):
                model = OilSpillModel(num_classes=1)
                model.load_state_dict(checkpoint, strict=False)
                model.eval()
                st.sidebar.success(f"✅ Model loaded!")
                return model
            elif isinstance(checkpoint, torch.nn.Module):
                checkpoint.eval()
                st.sidebar.success(f"✅ Full model loaded!")
                return checkpoint
                
        except Exception as e:
            st.sidebar.warning(f"⚠️ Failed: {str(e)[:100]}...")
            continue
    
    return None

# Try to load model
model = load_model()

# Enhanced demo model with realistic oil spill shapes
class RealisticDemoModel:
    def __init__(self):
        self.is_demo = True
    
    def __call__(self, x):
        batch, channels, height, width = x.shape
        output = torch.zeros(1, 1, height, width)
        
        # Create realistic oil spill shapes (more irregular and natural)
        y, x_coord = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        center_y, center_x = height // 2, width // 2
        
        # Main oil spill - irregular elliptical shape
        main_spill = self.create_irregular_shape(height, width, center_y, center_x, 
                                               main_radius_x=width//3, main_radius_y=height//4)
        output[0, 0, main_spill] = torch.rand(main_spill.sum()) * 0.4 + 0.6  # High confidence
        
        # Secondary spills - smaller irregular shapes
        for i in range(4):
            offset_y = torch.randint(-height//3, height//3, (1,))
            offset_x = torch.randint(-width//3, width//3, (1,))
            small_center_y, small_center_x = center_y + offset_y, center_x + offset_x
            
            small_spill = self.create_irregular_shape(height, width, small_center_y, small_center_x,
                                                    small_radius_x=width//10, small_radius_y=height//12)
            output[0, 0, small_spill] = torch.rand(small_spill.sum()) * 0.3 + 0.3  # Medium confidence
        
        # Oil sheens - very low confidence areas
        for i in range(8):
            sheen_y = torch.randint(0, height-5, (1,))
            sheen_x = torch.randint(0, width-5, (1,))
            sheen_h = torch.randint(5, 15, (1,))
            sheen_w = torch.randint(5, 15, (1,))
            
            sheen_mask = (x_coord >= sheen_x) & (x_coord < sheen_x + sheen_w) & \
                        (y >= sheen_y) & (y < sheen_y + sheen_h)
            output[0, 0, sheen_mask] = torch.rand(1) * 0.2 + 0.1  # Low confidence
        
        return torch.sigmoid(output)
    
    def create_irregular_shape(self, height, width, center_y, center_x, main_radius_x, main_radius_y):
        """Create irregular oil spill shapes"""
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        
        # Base elliptical shape
        base_shape = ((x - center_x)**2 / (main_radius_x)**2 + 
                     (y - center_y)**2 / (main_radius_y)**2 <= 1)
        
        # Add irregularity with noise
        noise = torch.rand(height, width) < 0.3  # 30% chance to remove pixels
        irregular_shape = base_shape & ~noise
        
        # Add some protrusions
        for angle in torch.linspace(0, 2*np.pi, 8):
            protrusion_x = center_x + (main_radius_x * 1.2) * torch.cos(angle)
            protrusion_y = center_y + (main_radius_y * 1.2) * torch.sin(angle)
            protrusion = ((x - protrusion_x)**2 / (main_radius_x//3)**2 + 
                         (y - protrusion_y)**2 / (main_radius_y//3)**2 <= 1)
            irregular_shape = irregular_shape | protrusion
        
        return irregular_shape

if model is None:
    st.info("🔸 **Demo Mode**: Showing realistic oil spill shapes. Upload your model for real detection.")
    model = RealisticDemoModel()

def extract_spill_shapes(mask, min_area=100):
    """Extract individual oil spill shapes with contours and properties"""
    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # Basic properties
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Shape analysis
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # Ellipse fitting
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (ellipse_x, ellipse_y), (major_axis, minor_axis), angle = ellipse
            eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0
        else:
            ellipse = None
            eccentricity = 0
        
        # Centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
        else:
            centroid_x, centroid_y = x + w//2, y + h//2
        
        shapes.append({
            'contour': contour,
            'hull': hull,
            'area': area,
            'perimeter': perimeter,
            'bbox': (x, y, w, h),
            'centroid': (centroid_x, centroid_y),
            'solidity': solidity,
            'circularity': circularity,
            'eccentricity': eccentricity,
            'ellipse': ellipse
        })
    
    # Sort by area (largest first)
    shapes.sort(key=lambda x: x['area'], reverse=True)
    return shapes

def create_shape_visualization(original_image, shapes, visualization_type='contours'):
    """Create different visualizations of oil spill shapes"""
    original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    result = original_cv.copy()
    
    if visualization_type == 'contours':
        # Draw contours only
        for shape in shapes:
            cv2.drawContours(result, [shape['contour']], -1, (0, 0, 255), 3)
            
    elif visualization_type == 'filled_contours':
        # Fill contours with transparent color
        overlay = result.copy()
        for shape in shapes:
            cv2.drawContours(overlay, [shape['contour']], -1, (0, 0, 255), -1)  # Fill
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
        # Draw outlines
        for shape in shapes:
            cv2.drawContours(result, [shape['contour']], -1, (0, 0, 255), 2)
            
    elif visualization_type == 'convex_hulls':
        # Show convex hulls
        for shape in shapes:
            cv2.drawContours(result, [shape['hull']], -1, (0, 255, 0), 2)  # Green hulls
            cv2.drawContours(result, [shape['contour']], -1, (0, 0, 255), 1)  # Red contours
            
    elif visualization_type == 'bounding_boxes':
        # Show bounding boxes with centroids
        for shape in shapes:
            x, y, w, h = shape['bbox']
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue boxes
            cv2.drawContours(result, [shape['contour']], -1, (0, 0, 255), 2)  # Red contours
            # Show centroid
            cx, cy = shape['centroid']
            cv2.circle(result, (cx, cy), 5, (255, 255, 0), -1)  # Yellow centroid
            
    elif visualization_type == 'ellipses':
        # Show fitted ellipses with orientation
        for shape in shapes:
            if shape['ellipse'] is not None:
                cv2.ellipse(result, shape['ellipse'], (255, 255, 0), 2)  # Cyan ellipses
            cv2.drawContours(result, [shape['contour']], -1, (0, 0, 255), 1)  # Red contours
            # Show centroid
            cx, cy = shape['centroid']
            cv2.circle(result, (cx, cy), 5, (255, 255, 0), -1)  # Yellow centroid
    
    elif visualization_type == 'shape_analysis':
        # Color code by shape properties
        for shape in shapes:
            # Color by circularity (red = circular, blue = irregular)
            circularity_color = int(shape['circularity'] * 255)
            color = (255 - circularity_color, 0, circularity_color)  # Blue to Red
            
            cv2.drawContours(result, [shape['contour']], -1, color, 3)
            
            # Label with area and centroid
            x, y, w, h = shape['bbox']
            area_text = f"{shape['area']:.0f}"
            cv2.putText(result, area_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Show centroid
            cx, cy = shape['centroid']
            cv2.circle(result, (cx, cy), 5, color, -1)
    
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result_rgb

def resize_contour(contour, target_width, target_height):
    """Resize contour to fit in target dimensions"""
    points = contour.reshape(-1, 2)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    
    width = x_max - x_min
    height = y_max - y_min
    
    if width == 0 or height == 0:
        return contour
    
    scale = min(target_width / width, target_height / height) * 0.8
    center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    
    new_points = ((points - np.array([center_x, center_y])) * scale + 
                 np.array([target_width / 2, target_height / 2]))
    
    return new_points.reshape(-1, 1, 2).astype(np.int32)

# Settings
st.sidebar.header("⚙️ Shape Detection Settings")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.01, 0.99, 0.5, 0.01)
min_spill_size = st.sidebar.slider("Minimum Spill Size", 10, 5000, 100, 10)

st.sidebar.subheader("Shape Visualization")
viz_type = st.sidebar.selectbox(
    "Visualization Type",
    ['contours', 'filled_contours', 'convex_hulls', 'bounding_boxes', 'ellipses', 'shape_analysis']
)

# Main application
st.header("📡 Upload Satellite Image")

uploaded_file = st.file_uploader("Choose satellite image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    original_size = image.size
    
    # Display layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🛰️ Original Image")
        st.image(image, use_container_width=True)
        st.write(f"Size: {original_size}")
    
    # Process image
    with st.spinner("🔍 Extracting oil spill shapes..."):
        try:
            # Simple preprocessing
            image_resized = image.resize((512, 512))
            img_array = np.array(image_resized) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = (img_array - mean) / std
            
            img_tensor = torch.from_numpy(img_array).float()
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                output = model(img_tensor)
            
            if isinstance(output, (list, tuple)):
                output = output[0]
            
            # Get probability map
            prob_map = output.squeeze().cpu().numpy()
            
            # Create initial mask
            binary_mask = (prob_map > confidence_threshold).astype(np.uint8) * 255
            binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
            
            # Extract shapes
            shapes = extract_spill_shapes(binary_mask_resized, min_spill_size)
            
            # Create shape visualizations
            shape_viz = create_shape_visualization(image, shapes, viz_type)
            
            # Display results
            with col2:
                st.subheader("🎭 Detected Shapes")
                st.image(shape_viz, use_container_width=True)
                st.write(f"**Found {len(shapes)} oil spill shapes**")
                st.write(f"Threshold: {confidence_threshold}")
                
                if hasattr(model, 'is_demo'):
                    st.caption("🔸 Demo shapes - adjust settings to see changes")
            
            with col3:
                st.subheader("📊 Shape Analysis")
                if shapes:
                    total_area = sum(shape['area'] for shape in shapes)
                    st.write(f"**Total Spill Area:** {total_area:,} pixels")
                    st.write(f"**Largest Spill:** {shapes[0]['area']:,} pixels")
                    st.write(f"**Number of Spills:** {len(shapes)}")
                    
                    # Show shape properties for largest spill
                    if shapes:
                        largest = shapes[0]
                        st.write("**Largest Spill Properties:**")
                        st.write(f"- Area: {largest['area']:,} px")
                        st.write(f"- Perimeter: {largest['perimeter']:.0f} px")
                        st.write(f"- Circularity: {largest['circularity']:.3f}")
                        st.write(f"- Solidity: {largest['solidity']:.3f}")
                        st.write(f"- Bounding Box: {largest['bbox'][2]}×{largest['bbox'][3]} px")
                else:
                    st.write("No shapes detected")
                    st.write("Try lowering the confidence threshold")
            
            # Individual shapes display
            if shapes:
                st.subheader("🔍 Individual Spill Shapes")
                
                # Show each shape with its properties
                for i, shape in enumerate(shapes[:6]):  # Show first 6 shapes
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Create individual shape visualization
                        individual_viz = np.zeros((200, 200, 3), dtype=np.uint8)
                        contour_resized = resize_contour(shape['contour'], 200, 200)
                        cv2.drawContours(individual_viz, [contour_resized], -1, (0, 0, 255), 2)
                        
                        st.image(individual_viz, use_container_width=True, caption=f"Shape {i+1}")
                    
                    with col2:
                        st.write(f"**Shape {i+1} Properties:**")
                        st.write(f"- Area: {shape['area']:,} pixels")
                        st.write(f"- Perimeter: {shape['perimeter']:.0f} pixels")
                        st.write(f"- Circularity: {shape['circularity']:.3f}")
                        st.write(f"- Solidity: {shape['solidity']:.3f}")
                        st.write(f"- Location: ({shape['centroid'][0]}, {shape['centroid'][1]})")
                
                if len(shapes) > 6:
                    st.info(f"📋 And {len(shapes) - 6} more shapes...")
            
            # Download options
            st.subheader("💾 Download Results")
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                # Download shape visualization
                viz_pil = Image.fromarray(shape_viz)
                buf_viz = io.BytesIO()
                viz_pil.save(buf_viz, format='PNG')
                st.download_button(
                    label="Download Shape Visualization",
                    data=buf_viz.getvalue(),
                    file_name="oil_spill_shapes.png",
                    mime="image/png"
                )
            
            with col_dl2:
                # Download shape data
                if shapes:
                    csv_data = "Shape,Area,Perimeter,Circularity,Solidity,Eccentricity,Centroid_X,Centroid_Y,Width,Height\n"
                    for i, shape in enumerate(shapes):
                        csv_data += f"{i+1},{shape['area']},{shape['perimeter']},{shape['circularity']:.3f},{shape['solidity']:.3f},{shape['eccentricity']:.3f},{shape['centroid'][0]},{shape['centroid'][1]},{shape['bbox'][2]},{shape['bbox'][3]}\n"
                    
                    st.download_button(
                        label="Download Shape Data (CSV)",
                        data=csv_data,
                        file_name="oil_spill_data.csv",
                        mime="text/csv"
                    )
            
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")

# File upload section
with st.expander("📤 Upload Your Model File"):
    st.markdown("""
    **To use your real model instead of demo mode:**
    """)
    
    uploaded_model = st.file_uploader(
        "Upload model file (.pth, .pt, .pkl)", 
        type=['pth', 'pt', 'pkl']
    )
    
    if uploaded_model is not None:
        try:
            model_filename = "uploaded_model.pth"
            with open(model_filename, "wb") as f:
                f.write(uploaded_model.getvalue())
            
            file_size = os.path.getsize(model_filename) / (1024 * 1024)
            st.success(f"✅ Model uploaded! ({file_size:.1f} MB)")
            st.info("🔄 **Refresh the page** to load your model")
            
        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")

# Quick start guide
with st.expander("🚀 Quick Start Guide"):
    st.markdown("""
    **To get the exact oil spill shapes:**
    
    1. **Upload any satellite image** - The demo will show realistic oil spill shapes
    2. **Adjust confidence threshold** - Lower for more detection, higher for less
    3. **Set minimum spill size** - Filter out small noisy detections
    4. **Choose visualization** - See contours, bounding boxes, or shape analysis
    5. **Download results** - Get the shape visualization and data
    
    **For real detection:**
    - Upload your model file using the section above
    - Refresh the page after upload
    - The app will automatically use your real model
    
    **Current files in directory:**
    """)
    
    files = os.listdir('.')
    for file in sorted(files):
        st.write(f"- `{file}`")
