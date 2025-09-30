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
    try:
        checkpoint = torch.load('oil_spill_model_deploy.pth', map_location='cpu')
        if isinstance(checkpoint, collections.OrderedDict):
            model = OilSpillModel(num_classes=1)
            model.load_state_dict(checkpoint, strict=False)
            model.eval()
            st.sidebar.success("✅ Model loaded")
            return model
    except Exception as e:
        st.sidebar.error(f"❌ Loading failed: {str(e)}")
        return None

model = load_model()

# Demo model for testing
class DemoModel:
    def __init__(self):
        self.is_demo = True
    
    def __call__(self, x):
        batch, channels, height, width = x.shape
        output = torch.zeros(1, 1, height, width)
        
        # Create realistic oil spill shapes
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        center_y, center_x = height // 2, width // 2
        
        # Main oil spill (irregular shape)
        main_spill = ((x - center_x)**2 / (width//3)**2 + (y - center_y)**2 / (height//4)**2 <= 1)
        output[0, 0, main_spill] = torch.rand(main_spill.sum()) * 0.4 + 0.6
        
        # Smaller spills around
        for i in range(3):
            offset_y = torch.randint(-height//3, height//3, (1,))
            offset_x = torch.randint(-width//3, width//3, (1,))
            small_center_y, small_center_x = center_y + offset_y, center_x + offset_x
            small_spill = ((x - small_center_x)**2 / (width//8)**2 + (y - small_center_y)**2 / (height//10)**2 <= 1)
            output[0, 0, small_spill] = torch.rand(small_spill.sum()) * 0.3 + 0.4
        
        return torch.sigmoid(output)

if model is None:
    st.warning("🔸 **Demo Mode**: Showing sample oil spill shapes")
    model = DemoModel()

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
        
        shapes.append({
            'contour': contour,
            'hull': hull,
            'area': area,
            'perimeter': perimeter,
            'bbox': (x, y, w, h),
            'solidity': solidity,
            'circularity': circularity,
            'eccentricity': eccentricity,
            'ellipse': ellipse
        })
    
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
        # Show bounding boxes
        for shape in shapes:
            x, y, w, h = shape['bbox']
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue boxes
            cv2.drawContours(result, [shape['contour']], -1, (0, 0, 255), 2)  # Red contours
            
    elif visualization_type == 'ellipses':
        # Show fitted ellipses
        for shape in shapes:
            if shape['ellipse'] is not None:
                cv2.ellipse(result, shape['ellipse'], (255, 255, 0), 2)  # Cyan ellipses
            cv2.drawContours(result, [shape['contour']], -1, (0, 0, 255), 1)  # Red contours
    
    elif visualization_type == 'shape_analysis':
        # Color code by shape properties
        for shape in shapes:
            # Color by circularity (red = circular, blue = irregular)
            circularity_color = int(shape['circularity'] * 255)
            color = (255 - circularity_color, 0, circularity_color)  # Blue to Red
            
            # Color by solidity (brighter = more solid)
            solidity_factor = int(shape['solidity'] * 200)
            color = tuple(min(255, c + solidity_factor) for c in color)
            
            cv2.drawContours(result, [shape['contour']], -1, color, 3)
            
            # Label with area
            x, y, w, h = shape['bbox']
            area_text = f"{shape['area']:.0f}"
            cv2.putText(result, area_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result_rgb

def create_shape_mask(shapes, image_size):
    """Create a clean mask with only the extracted shapes"""
    mask = np.zeros(image_size, dtype=np.uint8)
    for shape in shapes:
        cv2.drawContours(mask, [shape['contour']], -1, 255, -1)  # Fill contours
    return mask

# Settings
st.sidebar.header("⚙️ Shape Detection Settings")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.01, 0.99, 0.5, 0.01)
min_spill_size = st.sidebar.slider("Minimum Spill Size (pixels)", 10, 5000, 100, 10)

st.sidebar.subheader("Shape Visualization")
viz_type = st.sidebar.selectbox(
    "Visualization Type",
    ['contours', 'filled_contours', 'convex_hulls', 'bounding_boxes', 'ellipses', 'shape_analysis'],
    help="Different ways to visualize oil spill shapes"
)

show_shape_properties = st.sidebar.checkbox("Show Shape Properties", value=True)
extract_individual_shapes = st.sidebar.checkbox("Extract Individual Shapes", value=True)

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
    
    # Check if demo mode
    is_demo = hasattr(model, 'is_demo') and model.is_demo
    
    if is_demo:
        st.info("🔸 **Demo Mode**: Showing sample oil spill shapes")
    
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
            clean_shape_mask = create_shape_mask(shapes, binary_mask_resized.shape[:2])
            
            # Display results
            with col2:
                st.subheader("🎭 Detected Shapes")
                st.image(shape_viz, use_container_width=True)
                st.write(f"Found {len(shapes)} oil spill shapes")
                st.write(f"Threshold: {confidence_threshold}")
            
            with col3:
                st.subheader("🧹 Clean Shape Mask")
                st.image(clean_shape_mask, use_container_width=True)
                st.write("Binary mask of extracted shapes")
            
            # Shape analysis
            if shapes and show_shape_properties:
                st.subheader("📊 Shape Analysis")
                
                # Overall statistics
                total_area = sum(shape['area'] for shape in shapes)
                avg_circularity = np.mean([shape['circularity'] for shape in shapes])
                avg_solidity = np.mean([shape['solidity'] for shape in shapes])
                
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                
                with col_stats1:
                    st.metric("Total Shapes", len(shapes))
                with col_stats2:
                    st.metric("Total Area", f"{total_area:,} px")
                with col_stats3:
                    st.metric("Avg Circularity", f"{avg_circularity:.3f}")
                with col_stats4:
                    st.metric("Avg Solidity", f"{avg_solidity:.3f}")
                
                # Detailed shape properties
                st.write("**Individual Shape Properties:**")
                shape_data = []
                for i, shape in enumerate(shapes):
                    shape_data.append({
                        'Shape': i + 1,
                        'Area': int(shape['area']),
                        'Perimeter': int(shape['perimeter']),
                        'Circularity': f"{shape['circularity']:.3f}",
                        'Solidity': f"{shape['solidity']:.3f}",
                        'Eccentricity': f"{shape['eccentricity']:.3f}",
                        'BBox Size': f"{shape['bbox'][2]}x{shape['bbox'][3]}"
                    })
                
                st.table(shape_data)
            
            # Individual shape extraction
            if extract_individual_shapes and shapes:
                st.subheader("🔍 Individual Spill Shapes")
                
                # Show each shape separately
                num_shapes = len(shapes)
                cols_per_row = 4
                rows = (num_shapes + cols_per_row - 1) // cols_per_row
                
                for row in range(rows):
                    shape_cols = st.columns(cols_per_row)
                    for col_idx in range(cols_per_row):
                        shape_idx = row * cols_per_row + col_idx
                        if shape_idx < num_shapes:
                            with shape_cols[col_idx]:
                                # Create mask for individual shape
                                individual_mask = np.zeros_like(clean_shape_mask)
                                cv2.drawContours(individual_mask, [shapes[shape_idx]['contour']], -1, 255, -1)
                                
                                # Create overlay for individual shape
                                original_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                                overlay = original_cv.copy()
                                cv2.drawContours(overlay, [shapes[shape_idx]['contour']], -1, (0, 0, 255), 3)
                                
                                st.image(overlay, use_container_width=True, caption=f"Shape {shape_idx + 1}")
                                st.write(f"Area: {shapes[shape_idx]['area']:.0f} px")
            
            # Download options
            st.subheader("💾 Download Shape Data")
            
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            
            with col_dl1:
                # Download shape mask
                mask_pil = Image.fromarray(clean_shape_mask)
                buf_mask = io.BytesIO()
                mask_pil.save(buf_mask, format='PNG')
                st.download_button(
                    label="Download Shape Mask",
                    data=buf_mask.getvalue(),
                    file_name="oil_spill_shapes.png",
                    mime="image/png"
                )
            
            with col_dl2:
                # Download shape visualization
                viz_pil = Image.fromarray(shape_viz)
                buf_viz = io.BytesIO()
                viz_pil.save(buf_viz, format='PNG')
                st.download_button(
                    label="Download Shape Visualization",
                    data=buf_viz.getvalue(),
                    file_name="oil_spill_visualization.png",
                    mime="image/png"
                )
            
            with col_dl3:
                # Download shape data as CSV
                if shapes:
                    csv_data = "Shape,Area,Perimeter,Circularity,Solidity,Eccentricity,Width,Height\n"
                    for i, shape in enumerate(shapes):
                        csv_data += f"{i+1},{shape['area']},{shape['perimeter']},{shape['circularity']},{shape['solidity']},{shape['eccentricity']},{shape['bbox'][2]},{shape['bbox'][3]}\n"
                    
                    st.download_button(
                        label="Download Shape Data (CSV)",
                        data=csv_data,
                        file_name="oil_spill_shapes.csv",
                        mime="text/csv"
                    )
            
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")

# Shape analysis guide
with st.expander("🎯 Shape Analysis Guide"):
    st.markdown("""
    **Shape Properties Explained:**
    
    - **📍 Area**: Total pixels in the oil spill shape
    - **📏 Perimeter**: Length of the shape boundary
    - **⭕ Circularity**: How close to a perfect circle (1.0 = perfect circle)
    - **🔺 Solidity**: Ratio of area to convex hull area (1.0 = no concavities)
    - **🥚 Eccentricity**: How elongated the shape is (0.0 = circle, 1.0 = line)
    
    **Visualization Types:**
    - **Contours**: Red outlines around oil spills
    - **Filled Contours**: Transparent red fill with outlines
    - **Convex Hulls**: Green convex boundaries around shapes
    - **Bounding Boxes**: Blue rectangles around each spill
    - **Ellipses**: Cyan fitted ellipses showing orientation
    - **Shape Analysis**: Color-coded by shape properties
    
    **Tips for Better Shape Extraction:**
    - Adjust **confidence threshold** to capture spill boundaries accurately
    - Set **minimum spill size** to filter out noise
    - Use **individual shape extraction** to analyze each spill separately
    """)
