import cv2
import numpy as np
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

class AnnotationTool:
    def __init__(self):
        """
        Custom annotation tool for YOLOv5 dataset creation
        """
        self.root = tk.Tk()
        self.root.title("YOLOv5 Annotation Tool")
        self.root.geometry("1200x800")
        
        # Data storage
        self.current_image = None
        self.current_image_path = None
        self.annotations = []
        self.classes = []
        self.current_class = 0
        
        # Drawing state
        self.drawing = False
        self.start_point = None
        self.current_rect = None
        
        # Colors for different classes
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        # File operations
        file_frame = ttk.LabelFrame(control_frame, text="File Operations")
        file_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(file_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Save Annotations", command=self.save_annotations).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Load Dataset", command=self.load_dataset).pack(side=tk.LEFT, padx=5)
        
        # Class management
        class_frame = ttk.LabelFrame(control_frame, text="Classes")
        class_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        self.class_var = tk.StringVar()
        self.class_combo = ttk.Combobox(class_frame, textvariable=self.class_var, width=15)
        self.class_combo.pack(side=tk.LEFT, padx=5)
        self.class_combo.bind('<<ComboboxSelected>>', self.on_class_selected)
        
        ttk.Button(class_frame, text="Add Class", command=self.add_class).pack(side=tk.LEFT, padx=5)
        
        # Annotation controls
        annotation_frame = ttk.LabelFrame(control_frame, text="Annotations")
        annotation_frame.pack(side=tk.LEFT)
        
        ttk.Button(annotation_frame, text="Clear All", command=self.clear_annotations).pack(side=tk.LEFT, padx=5)
        ttk.Button(annotation_frame, text="Delete Last", command=self.delete_last_annotation).pack(side=tk.LEFT, padx=5)
        
        # Image display area
        self.canvas_frame = ttk.Frame(main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Load an image to start annotating")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        
        # Instructions
        instructions = """
        Instructions:
        1. Load an image or dataset
        2. Select/Add object class
        3. Click and drag to draw bounding boxes
        4. Right-click to delete a box
        5. Save annotations when done
        """
        
        ttk.Label(main_frame, text=instructions, justify=tk.LEFT).pack(side=tk.BOTTOM, anchor=tk.W)
        
    def load_image(self):
        """Load a single image for annotation"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.load_image_file(file_path)
    
    def load_image_file(self, file_path):
        """Load and display an image file"""
        self.current_image_path = file_path
        self.current_image = cv2.imread(file_path)
        
        if self.current_image is None:
            messagebox.showerror("Error", "Could not load image")
            return
        
        # Load existing annotations if they exist
        self.load_existing_annotations()
        
        # Display image
        self.display_image()
        
        self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
    
    def load_existing_annotations(self):
        """Load existing annotations for current image"""
        if not self.current_image_path:
            return
        
        # Look for YOLO format annotation file
        image_path = Path(self.current_image_path)
        annotation_path = image_path.parent / "labels" / f"{image_path.stem}.txt"
        
        self.annotations = []
        
        if annotation_path.exists():
            try:
                with open(annotation_path, 'r') as f:
                    lines = f.readlines()
                
                height, width = self.current_image.shape[:2]
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        box_width = float(parts[3]) * width
                        box_height = float(parts[4]) * height
                        
                        x1 = int(x_center - box_width / 2)
                        y1 = int(y_center - box_height / 2)
                        x2 = int(x_center + box_width / 2)
                        y2 = int(y_center + box_height / 2)
                        
                        self.annotations.append({
                            'class_id': class_id,
                            'bbox': (x1, y1, x2, y2)
                        })
                
                print(f"Loaded {len(self.annotations)} existing annotations")
                
            except Exception as e:
                print(f"Error loading annotations: {e}")
    
    def display_image(self):
        """Display the current image with annotations"""
        if self.current_image is None:
            return
        
        # Create a copy for display
        display_image = self.current_image.copy()
        
        # Draw existing annotations
        for i, annotation in enumerate(self.annotations):
            class_id = annotation['class_id']
            x1, y1, x2, y2 = annotation['bbox']
            
            # Get color for this class
            color = self.colors[class_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw class label
            class_name = self.classes[class_id] if class_id < len(self.classes) else f"Class_{class_id}"
            label = f"{class_name}"
            
            # Label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(display_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Resize image to fit display
        height, width = display_image.shape[:2]
        max_height = 600
        max_width = 800
        
        if height > max_height or width > max_width:
            scale = min(max_height / height, max_width / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_image = cv2.resize(display_image, (new_width, new_height))
        
        # Convert BGR to RGB for display
        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        
        # Create OpenCV window
        cv2.namedWindow("Annotation Tool", cv2.WINDOW_NORMAL)
        cv2.imshow("Annotation Tool", display_image)
        cv2.setMouseCallback("Annotation Tool", self.mouse_callback)
        
        # Update status
        self.status_var.set(f"Image: {os.path.basename(self.current_image_path)} | Annotations: {len(self.annotations)}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        if self.current_image is None:
            return
        
        # Scale coordinates back to original image size
        display_image = cv2.imread(self.current_image_path)
        original_height, original_width = display_image.shape[:2]
        
        # Get current display size
        current_display = cv2.getWindowImageRect("Annotation Tool")
        if current_display[2] > 0 and current_display[3] > 0:
            scale_x = original_width / current_display[2]
            scale_y = original_height / current_display[3]
            
            # Scale mouse coordinates
            x = int(x * scale_x)
            y = int(y * scale_y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Update current rectangle
            self.current_rect = (self.start_point[0], self.start_point[1], x, y)
            
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                self.drawing = False
                
                # Add annotation
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # Ensure proper rectangle
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Only add if rectangle is large enough
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    self.annotations.append({
                        'class_id': self.current_class,
                        'bbox': (x1, y1, x2, y2)
                    })
                    
                    print(f"Added annotation: Class {self.current_class}, BBox: ({x1}, {y1}, {x2}, {y2})")
                
                self.current_rect = None
                self.display_image()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Delete annotation at this point
            self.delete_annotation_at_point(x, y)
    
    def delete_annotation_at_point(self, x, y):
        """Delete annotation at the given point"""
        for i, annotation in enumerate(self.annotations):
            x1, y1, x2, y2 = annotation['bbox']
            if x1 <= x <= x2 and y1 <= y <= y2:
                del self.annotations[i]
                self.display_image()
                print(f"Deleted annotation {i}")
                break
    
    def add_class(self):
        """Add a new object class"""
        class_name = tk.simpledialog.askstring("Add Class", "Enter class name:")
        if class_name and class_name not in self.classes:
            self.classes.append(class_name)
            self.class_combo['values'] = self.classes
            self.class_combo.set(class_name)
            self.current_class = len(self.classes) - 1
            print(f"Added class: {class_name}")
    
    def on_class_selected(self, event):
        """Handle class selection"""
        selected_class = self.class_var.get()
        if selected_class in self.classes:
            self.current_class = self.classes.index(selected_class)
            print(f"Selected class: {selected_class} (ID: {self.current_class})")
    
    def save_annotations(self):
        """Save annotations in YOLO format"""
        if not self.current_image_path or not self.annotations:
            messagebox.showwarning("Warning", "No annotations to save")
            return
        
        # Create labels directory
        image_path = Path(self.current_image_path)
        labels_dir = image_path.parent / "labels"
        labels_dir.mkdir(exist_ok=True)
        
        # Save in YOLO format
        annotation_file = labels_dir / f"{image_path.stem}.txt"
        
        height, width = self.current_image.shape[:2]
        
        with open(annotation_file, 'w') as f:
            for annotation in self.annotations:
                class_id = annotation['class_id']
                x1, y1, x2, y2 = annotation['bbox']
                
                # Convert to YOLO format (normalized)
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
        
        # Save class names
        classes_file = image_path.parent / "classes.txt"
        with open(classes_file, 'w') as f:
            for class_name in self.classes:
                f.write(f"{class_name}\n")
        
        messagebox.showinfo("Success", f"Annotations saved to {annotation_file}")
        print(f"Saved {len(self.annotations)} annotations")
    
    def clear_annotations(self):
        """Clear all annotations"""
        self.annotations = []
        self.display_image()
        print("Cleared all annotations")
    
    def delete_last_annotation(self):
        """Delete the last annotation"""
        if self.annotations:
            deleted = self.annotations.pop()
            self.display_image()
            print(f"Deleted last annotation: {deleted}")
    
    def load_dataset(self):
        """Load a dataset directory for batch annotation"""
        dataset_dir = filedialog.askdirectory(title="Select Dataset Directory")
        if dataset_dir:
            self.load_dataset_directory(dataset_dir)
    
    def load_dataset_directory(self, dataset_dir):
        """Load all images from a dataset directory"""
        dataset_path = Path(dataset_dir)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(dataset_path.glob(f"**/*{ext}"))
        
        if not image_files:
            messagebox.showwarning("Warning", "No images found in dataset directory")
            return
        
        print(f"Found {len(image_files)} images in dataset")
        
        # Load first image
        if image_files:
            self.load_image_file(str(image_files[0]))
        
        # Store image list for navigation
        self.dataset_images = [str(img) for img in image_files]
        self.current_image_index = 0
        
        # Add navigation buttons
        self.add_navigation_controls()
    
    def add_navigation_controls(self):
        """Add navigation controls for dataset browsing"""
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        ttk.Button(nav_frame, text="Previous", command=self.previous_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        self.image_counter = tk.StringVar()
        ttk.Label(nav_frame, textvariable=self.image_counter).pack(side=tk.LEFT, padx=20)
        
        self.update_image_counter()
    
    def previous_image(self):
        """Load previous image in dataset"""
        if hasattr(self, 'dataset_images') and self.current_image_index > 0:
            self.save_annotations()  # Auto-save current annotations
            self.current_image_index -= 1
            self.load_image_file(self.dataset_images[self.current_image_index])
            self.update_image_counter()
    
    def next_image(self):
        """Load next image in dataset"""
        if hasattr(self, 'dataset_images') and self.current_image_index < len(self.dataset_images) - 1:
            self.save_annotations()  # Auto-save current annotations
            self.current_image_index += 1
            self.load_image_file(self.dataset_images[self.current_image_index])
            self.update_image_counter()
    
    def update_image_counter(self):
        """Update image counter display"""
        if hasattr(self, 'dataset_images'):
            total = len(self.dataset_images)
            current = self.current_image_index + 1
            self.image_counter.set(f"Image {current} of {total}")
    
    def run(self):
        """Start the annotation tool"""
        print("🏷️  Starting YOLOv5 Annotation Tool...")
        print("📋 Instructions:")
        print("  1. Load an image or dataset directory")
        print("  2. Add/select object classes")
        print("  3. Click and drag to draw bounding boxes")
        print("  4. Right-click on a box to delete it")
        print("  5. Save annotations when done")
        
        self.root.mainloop()

def main():
    """Run the annotation tool"""
    print("🚀 YOLOv5 Custom Annotation Tool")
    print("=" * 50)
    
    # Check if tkinter is available
    try:
        import tkinter.simpledialog
        tool = AnnotationTool()
        tool.run()
    except ImportError:
        print("❌ Tkinter not available. Using command-line annotation helper instead.")
        
        # Fallback: Simple command-line annotation helper
        print("\n💡 Command-line annotation helper:")
        print("1. Use LabelImg for GUI annotation: pip install labelImg")
        print("2. Or use online tools like Roboflow, CVAT, or LabelBox")
        print("3. Ensure annotations are in YOLO format:")
        print("   Format: class_id x_center y_center width height (normalized 0-1)")
        
        # Create sample annotation
        sample_annotation = """# Sample YOLO annotation file (image.txt)
# Format: class_id x_center y_center width height
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
"""
        print(f"\n📄 Sample annotation format:\n{sample_annotation}")

if __name__ == "__main__":
    main()
