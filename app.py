import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class OilSpillDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Oil Spill Detection System")
        self.root.geometry("1200x800")
        
        self.original_image = None
        self.processed_image = None
        self.detection_mask = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Detect Oil Spills", command=self.detect_oil).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save Result", command=self.save_result).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Advanced Analysis", command=self.advanced_analysis).pack(side=tk.LEFT, padx=5)
        
        # Parameters frame
        params_frame = ttk.Frame(control_frame)
        params_frame.pack(fill=tk.X, pady=10)
        
        # Sensitivity slider
        ttk.Label(params_frame, text="Detection Sensitivity:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.sensitivity_var = tk.DoubleVar(value=0.7)
        sensitivity_scale = ttk.Scale(params_frame, from_=0.1, to=1.0, variable=self.sensitivity_var, 
                                     orient=tk.HORIZONTAL)
        sensitivity_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
        
        # Min area slider
        ttk.Label(params_frame, text="Minimum Area:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.min_area_var = tk.IntVar(value=500)
        min_area_scale = ttk.Scale(params_frame, from_=100, to=5000, variable=self.min_area_var,
                                  orient=tk.HORIZONTAL)
        min_area_scale.grid(row=1, column=1, sticky=tk.EW, padx=5)
        
        params_frame.columnconfigure(1, weight=1)
        
        # Detection method
        ttk.Label(params_frame, text="Detection Method:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.method_var = tk.StringVar(value="advanced")
        methods = ttk.Combobox(params_frame, textvariable=self.method_var, 
                              values=["basic", "advanced", "kmeans", "combined"])
        methods.grid(row=2, column=1, sticky=tk.EW, padx=5)
        
        # Image display frame
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Original image
        self.original_frame = ttk.LabelFrame(display_frame, text="Original Image")
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.original_label = ttk.Label(self.original_frame)
        self.original_label.pack(padx=10, pady=10)
        
        # Processed image
        self.processed_frame = ttk.LabelFrame(display_frame, text="Oil Spill Detection Results")
        self.processed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.processed_label = ttk.Label(self.processed_frame)
        self.processed_label.pack(padx=10, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to load image")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise ValueError("Could not load image")
                
                self.display_image(self.original_image, self.original_label)
                self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
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
    
    def basic_color_detection(self, image, hsv, lab):
        """Basic color-based oil spill detection"""
        # HSV ranges for oil sheen (adjust based on your images)
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
    
    def advanced_edge_detection(self, image, gray):
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
    
    def detect_oil_spills_combined(self, image):
        """Combined detection method using multiple approaches"""
        # Preprocessing
        processed, hsv, lab = self.preprocess_image(image)
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Get masks from different methods
        color_mask = self.basic_color_detection(processed, hsv, lab)
        edge_mask = self.advanced_edge_detection(processed, gray)
        
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
        _, final_mask = cv2.threshold(combined_mask, 
                                    self.sensitivity_var.get() * 255, 
                                    255, cv2.THRESH_BINARY)
        
        final_mask = final_mask.astype(np.uint8)
        
        return final_mask
    
    def detect_oil(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        try:
            self.status_var.set("Detecting oil spills...")
            self.root.update()
            
            method = self.method_var.get()
            min_area = self.min_area_var.get()
            
            if method == "basic":
                processed, hsv, lab = self.preprocess_image(self.original_image)
                mask = self.basic_color_detection(processed, hsv, lab)
            elif method == "advanced":
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                mask = self.advanced_edge_detection(self.original_image, gray)
            elif method == "kmeans":
                segmented, labels = self.kmeans_segmentation(self.original_image)
                mask = np.zeros_like(self.original_image[:,:,0])
                # Select darker clusters as potential oil
                for label in np.unique(labels):
                    if np.mean(self.original_image[labels == label]) < 100:
                        mask[labels == label] = 255
            else:  # combined
                mask = self.detect_oil_spills_combined(self.original_image)
            
            # Refine the detection
            self.detection_mask = self.refine_detection_mask(mask, min_area)
            
            # Create visualization
            result_image = self.original_image.copy()
            result_image[self.detection_mask == 255] = [0, 0, 255]  # Red for oil
            
            # Add transparency for better visualization
            overlay = result_image.copy()
            cv2.addWeighted(overlay, 0.3, self.original_image, 0.7, 0, result_image)
            
            self.processed_image = result_image
            self.display_image(result_image, self.processed_label)
            
            # Calculate statistics
            total_pixels = self.original_image.shape[0] * self.original_image.shape[1]
            oil_pixels = np.sum(self.detection_mask == 255)
            oil_percentage = (oil_pixels / total_pixels) * 100
            
            self.status_var.set(
                f"Detection complete: {oil_percentage:.2f}% oil coverage "
                f"({oil_pixels} pixels) using {method} method"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            self.status_var.set("Detection failed")
    
    def advanced_analysis(self):
        if self.original_image is None or self.detection_mask is None:
            messagebox.showwarning("Warning", "Please detect oil spills first")
            return
        
        try:
            # Create analysis window
            analysis_window = tk.Toplevel(self.root)
            analysis_window.title("Advanced Analysis")
            analysis_window.geometry("1000x700")
            
            # Calculate detailed statistics
            contours, _ = cv2.findContours(self.detection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            analysis_text = f"Advanced Oil Spill Analysis\n"
            analysis_text += f"==========================\n\n"
            analysis_text += f"Total contaminated area: {np.sum(self.detection_mask == 255)} pixels\n"
            analysis_text += f"Number of separate oil spills: {len(contours)}\n\n"
            
            # Individual spill analysis
            analysis_text += "Individual Spill Analysis:\n"
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                
                analysis_text += f"Spill {i+1}: Area={area:.0f} px, Perimeter={perimeter:.0f} px, Circularity={circularity:.3f}\n"
            
            # Display analysis
            text_widget = tk.Text(analysis_window, wrap=tk.WORD)
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text_widget.insert(tk.END, analysis_text)
            
            # Add visualization
            viz_frame = ttk.Frame(analysis_window)
            viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create detailed visualization
            detailed_viz = self.original_image.copy()
            for i, contour in enumerate(contours):
                color = [np.random.randint(0, 255) for _ in range(3)]
                cv2.drawContours(detailed_viz, [contour], -1, color, 3)
                cv2.putText(detailed_viz, str(i+1), 
                           tuple(contour[0][0]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Convert and display
            detailed_viz_rgb = cv2.cvtColor(detailed_viz, cv2.COLOR_BGR2RGB)
            detailed_viz_pil = Image.fromarray(detailed_viz_rgb)
            detailed_viz_tk = ImageTk.PhotoImage(detailed_viz_pil.resize((400, 300)))
            
            viz_label = ttk.Label(viz_frame, image=detailed_viz_tk)
            viz_label.image = detailed_viz_tk
            viz_label.pack()
            
        except Exception as e:
            messagebox.showerror("Error", f"Advanced analysis failed: {str(e)}")
    
    def save_result(self):
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No processed image to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Result",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.processed_image)
                messagebox.showinfo("Success", f"Result saved to {file_path}")
                self.status_var.set(f"Result saved: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def display_image(self, image, label):
        """Display image in Tkinter label"""
        # Resize image to fit display
        h, w = image.shape[:2]
        max_size = 400
        
        if h > w:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w = max_size
            new_h = int(h * max_size / w)
        
        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil.resize((new_w, new_h)))
        
        label.configure(image=image_tk)
        label.image = image_tk

def main():
    root = tk.Tk()
    app = OilSpillDetector(root)
    root.mainloop()

if __name__ == "__main__":
    main()
