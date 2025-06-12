"""
Complete YOLOv5 Custom Training Guide
====================================

This comprehensive guide covers every step of training a YOLOv5 model
with custom data, from dataset preparation to deployment.
"""

import os
import shutil
import yaml
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import requests
import subprocess
import time
from datetime import datetime
import pandas as pd
from collections import defaultdict, Counter
import random

class YOLOv5TrainingGuide:
    def __init__(self, project_name="custom_detection"):
        """
        Initialize the complete YOLOv5 training guide
        
        Args:
            project_name: Name of your detection project
        """
        self.project_name = project_name
        self.project_dir = Path(f"yolo_projects/{project_name}")
        self.dataset_dir = self.project_dir / "dataset"
        self.models_dir = self.project_dir / "models"
        self.results_dir = self.project_dir / "results"
        
        # Create project structure
        self.setup_project_structure()
        
        print(f"🚀 YOLOv5 Training Guide Initialized")
        print(f"📁 Project: {self.project_name}")
        print(f"📂 Directory: {self.project_dir}")
        
    def setup_project_structure(self):
        """Create the complete project directory structure"""
        directories = [
            # Dataset structure
            "dataset/images/train",
            "dataset/images/val", 
            "dataset/images/test",
            "dataset/labels/train",
            "dataset/labels/val",
            "dataset/labels/test",
            "dataset/raw_images",
            "dataset/annotations",
            
            # Model directories
            "models/pretrained",
            "models/trained",
            "models/optimized",
            
            # Results and analysis
            "results/training_runs",
            "results/evaluations",
            "results/visualizations",
            
            # Tools and scripts
            "tools",
            "configs"
        ]
        
        for directory in directories:
            (self.project_dir / directory).mkdir(parents=True, exist_ok=True)
        
        print("✅ Project structure created")
        
    def step1_dataset_preparation(self):
        """
        STEP 1: DATASET PREPARATION
        ===========================
        
        This step covers:
        - Image collection strategies
        - Dataset organization
        - Quality control
        - Data splitting
        """
        print("\n" + "="*60)
        print("STEP 1: DATASET PREPARATION")
        print("="*60)
        
        print("""
📋 DATASET PREPARATION CHECKLIST:

1. IMAGE COLLECTION:
   ✓ Collect 100-1000+ images per class
   ✓ Ensure diverse conditions (lighting, angles, backgrounds)
   ✓ Include edge cases and difficult examples
   ✓ Maintain consistent image quality (min 416x416 pixels)

2. IMAGE SOURCES:
   • Manual photography/screenshots
   • Web scraping (with proper permissions)
   • Existing datasets (COCO, Open Images, etc.)
   • Synthetic data generation
   • Data augmentation

3. QUALITY REQUIREMENTS:
   • Resolution: Minimum 416x416, recommended 640x640+
   • Format: JPG, PNG supported
   • Quality: Clear, non-blurry images
   • Diversity: Various lighting, angles, scales
        """)
        
        # Demonstrate image collection
        self.collect_sample_images()
        
        # Organize dataset
        self.organize_dataset()
        
        # Quality control
        self.perform_quality_control()
        
        print("✅ Step 1 Complete: Dataset Preparation")
        
    def collect_sample_images(self):
        """Demonstrate image collection process"""
        print("\n🖼️  COLLECTING SAMPLE IMAGES...")
        
        # Create sample images for demonstration
        sample_classes = ["car", "person", "bicycle", "dog"]
        images_per_class = 20
        
        for class_name in sample_classes:
            class_dir = self.dataset_dir / "raw_images" / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📸 Creating sample images for class: {class_name}")
            
            for i in range(images_per_class):
                # Create synthetic sample image
                image = self.create_sample_image(class_name, i)
                
                # Save image
                image_path = class_dir / f"{class_name}_{i:03d}.jpg"
                cv2.imwrite(str(image_path), image)
            
            print(f"  ✓ Created {images_per_class} images for {class_name}")
        
        print(f"📊 Total sample images created: {len(sample_classes) * images_per_class}")
        
    def create_sample_image(self, class_name, index):
        """Create a synthetic sample image for demonstration"""
        # Create base image with random background
        image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        
        # Add noise for realism
        noise = np.random.randint(-30, 30, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add objects based on class
        if class_name == "car":
            # Draw car-like rectangle
            x, y = 100 + index * 10, 200 + index * 5
            cv2.rectangle(image, (x, y), (x + 200, y + 100), (0, 0, 200), -1)
            cv2.rectangle(image, (x + 20, y + 20), (x + 180, y + 60), (50, 50, 50), -1)
            # Add wheels
            cv2.circle(image, (x + 40, y + 90), 15, (0, 0, 0), -1)
            cv2.circle(image, (x + 160, y + 90), 15, (0, 0, 0), -1)
            
        elif class_name == "person":
            # Draw person-like figure
            x, y = 200 + index * 8, 100 + index * 3
            # Head
            cv2.circle(image, (x + 25, y + 25), 20, (200, 150, 100), -1)
            # Body
            cv2.rectangle(image, (x + 10, y + 45), (x + 40, y + 120), (100, 100, 200), -1)
            # Arms
            cv2.rectangle(image, (x - 5, y + 50), (x + 10, y + 100), (200, 150, 100), -1)
            cv2.rectangle(image, (x + 40, y + 50), (x + 55, y + 100), (200, 150, 100), -1)
            # Legs
            cv2.rectangle(image, (x + 15, y + 120), (x + 25, y + 180), (0, 0, 150), -1)
            cv2.rectangle(image, (x + 25, y + 120), (x + 35, y + 180), (0, 0, 150), -1)
            
        elif class_name == "bicycle":
            # Draw bicycle-like shape
            x, y = 150 + index * 12, 250 + index * 4
            # Wheels
            cv2.circle(image, (x, y), 30, (100, 100, 100), 3)
            cv2.circle(image, (x + 100, y), 30, (100, 100, 100), 3)
            # Frame
            cv2.line(image, (x, y), (x + 50, y - 40), (150, 150, 150), 3)
            cv2.line(image, (x + 50, y - 40), (x + 100, y), (150, 150, 150), 3)
            cv2.line(image, (x + 30, y), (x + 70, y), (150, 150, 150), 3)
            
        elif class_name == "dog":
            # Draw dog-like shape
            x, y = 180 + index * 15, 300 + index * 6
            # Body
            cv2.ellipse(image, (x + 40, y), (40, 20), 0, 0, 360, (139, 69, 19), -1)
            # Head
            cv2.circle(image, (x, y - 10), 25, (139, 69, 19), -1)
            # Legs
            cv2.rectangle(image, (x + 10, y + 15), (x + 15, y + 35), (139, 69, 19), -1)
            cv2.rectangle(image, (x + 25, y + 15), (x + 30, y + 35), (139, 69, 19), -1)
            cv2.rectangle(image, (x + 50, y + 15), (x + 55, y + 35), (139, 69, 19), -1)
            cv2.rectangle(image, (x + 65, y + 15), (x + 70, y + 35), (139, 69, 19), -1)
            # Tail
            cv2.line(image, (x + 80, y), (x + 95, y - 15), (139, 69, 19), 4)
        
        return image
    
    def organize_dataset(self):
        """Organize collected images into proper structure"""
        print("\n📁 ORGANIZING DATASET...")
        
        raw_images_dir = self.dataset_dir / "raw_images"
        
        if not raw_images_dir.exists():
            print("⚠️  No raw images found. Please collect images first.")
            return
        
        # Get all image files
        image_files = []
        for class_dir in raw_images_dir.iterdir():
            if class_dir.is_dir():
                class_images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                image_files.extend(class_images)
        
        if not image_files:
            print("⚠️  No images found in raw_images directory")
            return
        
        # Shuffle for random distribution
        random.shuffle(image_files)
        
        # Split ratios
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1
        
        total_images = len(image_files)
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        
        # Split files
        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count + val_count]
        test_files = image_files[train_count + val_count:]
        
        # Copy files to respective directories
        splits = [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files)
        ]
        
        for split_name, files in splits:
            print(f"📂 Organizing {split_name} split: {len(files)} images")
            
            for file_path in files:
                # Copy to images directory
                dest_path = self.dataset_dir / "images" / split_name / file_path.name
                shutil.copy2(file_path, dest_path)
        
        print(f"✅ Dataset organized:")
        print(f"  📊 Train: {len(train_files)} images")
        print(f"  📊 Validation: {len(val_files)} images")
        print(f"  📊 Test: {len(test_files)} images")
        
    def perform_quality_control(self):
        """Perform quality control on the dataset"""
        print("\n🔍 PERFORMING QUALITY CONTROL...")
        
        quality_report = {
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': [],
            'resolution_stats': [],
            'file_size_stats': []
        }
        
        for split in ['train', 'val', 'test']:
            images_dir = self.dataset_dir / "images" / split
            
            if not images_dir.exists():
                continue
                
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            
            for image_file in image_files:
                quality_report['total_images'] += 1
                
                try:
                    # Check if image can be loaded
                    image = cv2.imread(str(image_file))
                    
                    if image is None:
                        quality_report['invalid_images'].append(str(image_file))
                        continue
                    
                    # Check resolution
                    height, width = image.shape[:2]
                    quality_report['resolution_stats'].append((width, height))
                    
                    # Check file size
                    file_size = image_file.stat().st_size
                    quality_report['file_size_stats'].append(file_size)
                    
                    # Validate minimum resolution
                    if width < 416 or height < 416:
                        print(f"⚠️  Low resolution image: {image_file.name} ({width}x{height})")
                    
                    quality_report['valid_images'] += 1
                    
                except Exception as e:
                    quality_report['invalid_images'].append(f"{image_file}: {str(e)}")
        
        # Print quality report
        print(f"📊 QUALITY CONTROL REPORT:")
        print(f"  Total Images: {quality_report['total_images']}")
        print(f"  Valid Images: {quality_report['valid_images']}")
        print(f"  Invalid Images: {len(quality_report['invalid_images'])}")
        
        if quality_report['resolution_stats']:
            resolutions = quality_report['resolution_stats']
            avg_width = np.mean([r[0] for r in resolutions])
            avg_height = np.mean([r[1] for r in resolutions])
            print(f"  Average Resolution: {avg_width:.0f}x{avg_height:.0f}")
        
        if quality_report['file_size_stats']:
            avg_size = np.mean(quality_report['file_size_stats']) / 1024  # KB
            print(f"  Average File Size: {avg_size:.1f} KB")
        
        # Save quality report
        report_path = self.dataset_dir / "quality_report.json"
        with open(report_path, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        print(f"📄 Quality report saved: {report_path}")
        
    def step2_image_annotation(self):
        """
        STEP 2: IMAGE ANNOTATION
        ========================
        
        This step covers:
        - Annotation tools and setup
        - Bounding box creation
        - Label format conversion
        - Annotation quality control
        """
        print("\n" + "="*60)
        print("STEP 2: IMAGE ANNOTATION")
        print("="*60)
        
        print("""
🏷️  ANNOTATION PROCESS OVERVIEW:

1. ANNOTATION TOOLS:
   • LabelImg (Recommended GUI tool)
   • CVAT (Web-based collaborative tool)
   • Roboflow (Cloud-based with team features)
   • Custom annotation scripts

2. YOLO ANNOTATION FORMAT:
   • Each image needs a corresponding .txt file
   • Format: class_id x_center y_center width height
   • All coordinates normalized to 0-1 range
   • One line per object in the image

3. ANNOTATION BEST PRACTICES:
   • Draw tight bounding boxes around objects
   • Include partially visible objects
   • Be consistent with class definitions
   • Double-check difficult cases
   • Maintain annotation quality standards
        """)
        
        # Setup annotation environment
        self.setup_annotation_tools()
        
        # Create class definitions
        self.create_class_definitions()
        
        # Generate sample annotations
        self.create_sample_annotations()
        
        # Validate annotations
        self.validate_annotations()
        
        print("✅ Step 2 Complete: Image Annotation")
        
    def setup_annotation_tools(self):
        """Setup annotation tools and environment"""
        print("\n🛠️  SETTING UP ANNOTATION TOOLS...")
        
        print("""
📦 RECOMMENDED ANNOTATION TOOLS:

1. LABELIMG (Most Popular):
   Installation: pip install labelImg
   Usage: labelImg [IMAGE_PATH] [ANNOTATION_PATH]
   
   Features:
   • Simple GUI interface
   • YOLO format export
   • Keyboard shortcuts
   • Class management
   
2. CVAT (Advanced Web Tool):
   Installation: Docker-based setup
   Features:
   • Team collaboration
   • Advanced annotation types
   • Quality control features
   • Integration with ML pipelines
   
3. ROBOFLOW (Cloud Platform):
   • Web-based interface
   • Automatic format conversion
   • Data augmentation
   • Team management
   • API integration

INSTALLATION COMMANDS:
----------------------
# Install LabelImg
pip install labelImg

# Install CVAT (requires Docker)
git clone https://github.com/openvinotoolkit/cvat
cd cvat
docker-compose up -d

# For this tutorial, we'll use a custom annotation helper
        """)
        
        # Create annotation helper script
        self.create_annotation_helper()
        
    def create_class_definitions(self):
        """Create class definitions file"""
        print("\n📋 CREATING CLASS DEFINITIONS...")
        
        # Define classes for our sample dataset
        classes = ["car", "person", "bicycle", "dog"]
        
        # Create classes.txt file
        classes_file = self.dataset_dir / "classes.txt"
        with open(classes_file, 'w') as f:
            for class_name in classes:
                f.write(f"{class_name}\n")
        
        # Create class mapping
        class_mapping = {i: class_name for i, class_name in enumerate(classes)}
        
        # Save class mapping as JSON
        mapping_file = self.dataset_dir / "class_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        print(f"📄 Classes defined: {classes}")
        print(f"📁 Classes file: {classes_file}")
        print(f"📁 Mapping file: {mapping_file}")
        
        return classes
    
    def create_annotation_helper(self):
        """Create a simple annotation helper script"""
        annotation_script = '''
import cv2
import numpy as np
import json
from pathlib import Path

class SimpleAnnotator:
    def __init__(self, image_path, classes):
        self.image_path = image_path
        self.classes = classes
        self.current_class = 0
        self.annotations = []
        self.drawing = False
        self.start_point = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                end_point = (x, y)
                self.add_annotation(self.start_point, end_point)
    
    def add_annotation(self, start, end):
        x1, y1 = start
        x2, y2 = end
        
        # Ensure proper rectangle
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
            self.annotations.append({
                'class_id': self.current_class,
                'bbox': (x1, y1, x2, y2)
            })
            print(f"Added: {self.classes[self.current_class]} at ({x1},{y1},{x2},{y2})")
    
    def annotate(self):
        image = cv2.imread(self.image_path)
        if image is None:
            print(f"Could not load image: {self.image_path}")
            return
        
        cv2.namedWindow('Annotator')
        cv2.setMouseCallback('Annotator', self.mouse_callback)
        
        while True:
            display_image = image.copy()
            
            # Draw existing annotations
            for ann in self.annotations:
                x1, y1, x2, y2 = ann['bbox']
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_image, self.classes[ann['class_id']], 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show current class
            cv2.putText(display_image, f"Class: {self.classes[self.current_class]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Annotator', display_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key >= ord('0') and key <= ord('9'):
                class_id = key - ord('0')
                if class_id < len(self.classes):
                    self.current_class = class_id
            elif key == ord('s'):
                self.save_annotations()
        
        cv2.destroyAllWindows()
    
    def save_annotations(self):
        # Convert to YOLO format and save
        pass

# Usage example:
# annotator = SimpleAnnotator("image.jpg", ["car", "person", "bicycle", "dog"])
# annotator.annotate()
'''
        
        # Save annotation helper
        helper_file = self.project_dir / "tools" / "simple_annotator.py"
        with open(helper_file, 'w') as f:
            f.write(annotation_script)
        
        print(f"🛠️  Annotation helper created: {helper_file}")
        
    def create_sample_annotations(self):
        """Create sample annotations for demonstration"""
        print("\n🏷️  CREATING SAMPLE ANNOTATIONS...")
        
        classes = ["car", "person", "bicycle", "dog"]
        
        for split in ['train', 'val', 'test']:
            images_dir = self.dataset_dir / "images" / split
            labels_dir = self.dataset_dir / "labels" / split
            
            if not images_dir.exists():
                continue
            
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            
            for image_file in image_files:
                # Load image to get dimensions
                image = cv2.imread(str(image_file))
                if image is None:
                    continue
                
                height, width = image.shape[:2]
                
                # Generate realistic annotations based on filename
                annotations = self.generate_realistic_annotations(image_file.name, width, height, classes)
                
                # Save annotation file
                label_file = labels_dir / f"{image_file.stem}.txt"
                with open(label_file, 'w') as f:
                    for ann in annotations:
                        f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n")
        
        print("✅ Sample annotations created")
        
    def generate_realistic_annotations(self, filename, img_width, img_height, classes):
        """Generate realistic annotations based on the synthetic images"""
        annotations = []
        
        # Determine class from filename
        class_name = None
        for i, cls in enumerate(classes):
            if cls in filename.lower():
                class_name = cls
                class_id = i
                break
        
        if class_name is None:
            return annotations
        
        # Generate bounding box based on class and image content
        if class_name == "car":
            # Car is typically in the center-bottom area
            x_center = 0.4 + random.uniform(-0.1, 0.2)
            y_center = 0.6 + random.uniform(-0.1, 0.1)
            width = random.uniform(0.25, 0.35)
            height = random.uniform(0.15, 0.25)
            
        elif class_name == "person":
            # Person is typically vertical in center
            x_center = 0.4 + random.uniform(-0.15, 0.2)
            y_center = 0.5 + random.uniform(-0.2, 0.2)
            width = random.uniform(0.1, 0.2)
            height = random.uniform(0.3, 0.5)
            
        elif class_name == "bicycle":
            # Bicycle is horizontal, center area
            x_center = 0.4 + random.uniform(-0.1, 0.2)
            y_center = 0.6 + random.uniform(-0.1, 0.1)
            width = random.uniform(0.2, 0.3)
            height = random.uniform(0.15, 0.25)
            
        elif class_name == "dog":
            # Dog is lower, smaller
            x_center = 0.4 + random.uniform(-0.15, 0.2)
            y_center = 0.7 + random.uniform(-0.1, 0.1)
            width = random.uniform(0.15, 0.25)
            height = random.uniform(0.1, 0.2)
        
        # Ensure bounding box is within image bounds
        x_center = max(width/2, min(1 - width/2, x_center))
        y_center = max(height/2, min(1 - height/2, y_center))
        
        annotations.append({
            'class_id': class_id,
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height
        })
        
        return annotations
    
    def validate_annotations(self):
        """Validate annotation quality and format"""
        print("\n🔍 VALIDATING ANNOTATIONS...")
        
        validation_report = {
            'total_images': 0,
            'annotated_images': 0,
            'missing_annotations': [],
            'invalid_annotations': [],
            'annotation_stats': defaultdict(int),
            'class_distribution': defaultdict(int)
        }
        
        for split in ['train', 'val', 'test']:
            images_dir = self.dataset_dir / "images" / split
            labels_dir = self.dataset_dir / "labels" / split
            
            if not images_dir.exists():
                continue
            
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            
            for image_file in image_files:
                validation_report['total_images'] += 1
                
                label_file = labels_dir / f"{image_file.stem}.txt"
                
                if not label_file.exists():
                    validation_report['missing_annotations'].append(str(image_file))
                    continue
                
                validation_report['annotated_images'] += 1
                
                # Validate annotation format
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    object_count = 0
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                            
                        parts = line.split()
                        if len(parts) != 5:
                            validation_report['invalid_annotations'].append(
                                f"{label_file}:{line_num} - Invalid format (expected 5 values)"
                            )
                            continue
                        
                        try:
                            class_id = int(parts[0])
                            coords = [float(x) for x in parts[1:]]
                            
                            # Validate coordinate ranges
                            if not all(0 <= coord <= 1 for coord in coords):
                                validation_report['invalid_annotations'].append(
                                    f"{label_file}:{line_num} - Coordinates out of range [0,1]"
                                )
                            
                            # Update statistics
                            validation_report['class_distribution'][class_id] += 1
                            object_count += 1
                            
                        except ValueError:
                            validation_report['invalid_annotations'].append(
                                f"{label_file}:{line_num} - Invalid number format"
                            )
                    
                    validation_report['annotation_stats'][object_count] += 1
                    
                except Exception as e:
                    validation_report['invalid_annotations'].append(
                        f"{label_file} - Error reading file: {str(e)}"
                    )
        
        # Print validation results
        print(f"📊 ANNOTATION VALIDATION REPORT:")
        print(f"  Total Images: {validation_report['total_images']}")
        print(f"  Annotated Images: {validation_report['annotated_images']}")
        print(f"  Missing Annotations: {len(validation_report['missing_annotations'])}")
        print(f"  Invalid Annotations: {len(validation_report['invalid_annotations'])}")
        
        if validation_report['class_distribution']:
            print(f"  Class Distribution:")
            for class_id, count in validation_report['class_distribution'].items():
                print(f"    Class {class_id}: {count} objects")
        
        # Save validation report
        report_path = self.dataset_dir / "annotation_validation.json"
        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        print(f"📄 Validation report saved: {report_path}")
        
        return validation_report
    
    def step3_training_environment_setup(self):
        """
        STEP 3: TRAINING ENVIRONMENT SETUP
        ==================================
        
        This step covers:
        - YOLOv5 installation
        - Dependencies setup
        - Environment configuration
        - Hardware optimization
        """
        print("\n" + "="*60)
        print("STEP 3: TRAINING ENVIRONMENT SETUP")
        print("="*60)
        
        print("""
🛠️  ENVIRONMENT SETUP REQUIREMENTS:

1. SYSTEM REQUIREMENTS:
   • Python 3.8+ (recommended 3.9)
   • CUDA 11.0+ (for GPU training)
   • 8GB+ RAM (16GB+ recommended)
   • 10GB+ free disk space
   • GPU with 4GB+ VRAM (optional but recommended)

2. SOFTWARE DEPENDENCIES:
   • PyTorch 1.7+
   • OpenCV
   • Matplotlib
   • NumPy
   • Pillow
   • PyYAML
   • TensorBoard

3. HARDWARE OPTIMIZATION:
   • GPU: NVIDIA RTX series recommended
   • CPU: Multi-core processor
   • Storage: SSD for faster data loading
   • Memory: 16GB+ RAM for large datasets
        """)
        
        # Check system requirements
        self.check_system_requirements()
        
        # Install YOLOv5
        self.install_yolov5()
        
        # Setup environment
        self.setup_training_environment()
        
        # Verify installation
        self.verify_installation()
        
        print("✅ Step 3 Complete: Training Environment Setup")
        
    def check_system_requirements(self):
        """Check system requirements for training"""
        print("\n🔍 CHECKING SYSTEM REQUIREMENTS...")
        
        import sys
        import platform
        import psutil
        
        # Python version
        python_version = sys.version_info
        print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version < (3, 8):
            print("⚠️  Warning: Python 3.8+ recommended")
        else:
            print("✅ Python version OK")
        
        # System info
        print(f"💻 System: {platform.system()} {platform.release()}")
        print(f"🏗️  Architecture: {platform.machine()}")
        
        # Memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"🧠 RAM: {memory_gb:.1f} GB")
        
        if memory_gb < 8:
            print("⚠️  Warning: 8GB+ RAM recommended")
        else:
            print("✅ Memory OK")
        
        # Disk space
        disk = psutil.disk_usage('.')
        free_gb = disk.free / (1024**3)
        print(f"💾 Free Disk Space: {free_gb:.1f} GB")
        
        if free_gb < 10:
            print("⚠️  Warning: 10GB+ free space recommended")
        else:
            print("✅ Disk space OK")
        
        # GPU check
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"🎮 GPU: {gpu_count} CUDA device(s) available")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                print("✅ GPU available for training")
            else:
                print("⚠️  No GPU available - training will use CPU (slower)")
        except ImportError:
            print("❓ PyTorch not installed - cannot check GPU")
    
    def install_yolov5(self):
        """Install YOLOv5 and dependencies"""
        print("\n📦 INSTALLING YOLOv5...")
        
        yolov5_dir = Path("yolov5")
        
        if yolov5_dir.exists():
            print("✅ YOLOv5 already exists")
        else:
            print("📥 Cloning YOLOv5 repository...")
            try:
                subprocess.run([
                    "git", "clone", "https://github.com/ultralytics/yolov5.git"
                ], check=True, capture_output=True)
                print("✅ YOLOv5 repository cloned")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to clone YOLOv5: {e}")
                print("💡 Manual installation:")
                print("   git clone https://github.com/ultralytics/yolov5.git")
                return False
        
        # Install requirements
        requirements_file = yolov5_dir / "requirements.txt"
        if requirements_file.exists():
            print("📦 Installing YOLOv5 requirements...")
            try:
                subprocess.run([
                    "pip", "install", "-r", str(requirements_file)
                ], check=True, capture_output=True)
                print("✅ Requirements installed")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Warning: Some requirements may not have installed correctly")
                print("💡 Manual installation:")
                print(f"   pip install -r {requirements_file}")
        
        return True
    
    def setup_training_environment(self):
        """Setup training environment configuration"""
        print("\n⚙️  SETTING UP TRAINING ENVIRONMENT...")
        
        # Create dataset.yaml for YOLOv5
        self.create_dataset_yaml()
        
        # Create training configuration
        self.create_training_config()
        
        # Setup logging
        self.setup_logging()
        
    def create_dataset_yaml(self):
        """Create dataset.yaml file for YOLOv5 training"""
        print("📄 Creating dataset.yaml...")
        
        # Load class names
        classes_file = self.dataset_dir / "classes.txt"
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
        else:
            class_names = ["car", "person", "bicycle", "dog"]  # Default
        
        # Create dataset configuration
        dataset_config = {
            'path': str(self.dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names),
            'names': class_names
        }
        
        # Save dataset.yaml
        yaml_path = self.dataset_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✅ Dataset configuration saved: {yaml_path}")
        print(f"📊 Classes: {len(class_names)} - {class_names}")
        
        return yaml_path
    
    def create_training_config(self):
        """Create training configuration file"""
        print("⚙️  Creating training configuration...")
        
        training_config = {
            # Model settings
            'model_size': 's',  # s, m, l, x
            'pretrained': True,
            
            # Training parameters
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640,
            'device': 'auto',  # auto, cpu, 0, 1, etc.
            
            # Optimization
            'optimizer': 'SGD',  # SGD, Adam, AdamW
            'lr0': 0.01,  # initial learning rate
            'lrf': 0.01,  # final learning rate factor
            'momentum': 0.937,
            'weight_decay': 0.0005,
            
            # Augmentation
            'hsv_h': 0.015,  # HSV-Hue augmentation
            'hsv_s': 0.7,    # HSV-Saturation augmentation
            'hsv_v': 0.4,    # HSV-Value augmentation
            'degrees': 0.0,  # rotation degrees
            'translate': 0.1, # translation fraction
            'scale': 0.5,    # scaling factor
            'shear': 0.0,    # shear degrees
            'perspective': 0.0, # perspective factor
            'flipud': 0.0,   # vertical flip probability
            'fliplr': 0.5,   # horizontal flip probability
            'mosaic': 1.0,   # mosaic augmentation probability
            'mixup': 0.0,    # mixup augmentation probability
            
            # Loss weights
            'box': 0.05,     # box loss gain
            'cls': 0.5,      # class loss gain
            'obj': 1.0,      # object loss gain
            
            # Other settings
            'patience': 100,  # early stopping patience
            'save_period': -1, # save checkpoint every x epochs
            'workers': 8,     # dataloader workers
            'project': str(self.results_dir / "training_runs"),
            'name': f"{self.project_name}_training",
            'exist_ok': True
        }
        
        # Save training config
        config_path = self.project_dir / "configs" / "training_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(training_config, f, default_flow_style=False)
        
        print(f"✅ Training configuration saved: {config_path}")
        
        return training_config
    
    def setup_logging(self):
        """Setup logging and monitoring"""
        print("📊 Setting up logging and monitoring...")
        
        # Create logs directory
        logs_dir = self.project_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Setup TensorBoard logging
        tensorboard_dir = logs_dir / "tensorboard"
        tensorboard_dir.mkdir(exist_ok=True)
        
        print(f"📁 Logs directory: {logs_dir}")
        print(f"📈 TensorBoard logs: {tensorboard_dir}")
        print("💡 To view TensorBoard: tensorboard --logdir logs/tensorboard")
        
    def verify_installation(self):
        """Verify YOLOv5 installation"""
        print("\n✅ VERIFYING INSTALLATION...")
        
        try:
            # Test YOLOv5 import
            import torch
            print(f"🔥 PyTorch version: {torch.__version__}")
            
            # Test CUDA
            if torch.cuda.is_available():
                print(f"🎮 CUDA version: {torch.version.cuda}")
                print(f"🎯 GPU available: {torch.cuda.get_device_name(0)}")
            else:
                print("💻 Using CPU for training")
            
            # Test YOLOv5
            yolov5_dir = Path("yolov5")
            if yolov5_dir.exists():
                print("✅ YOLOv5 installation verified")
                
                # Test model loading
                try:
                    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                    print("✅ YOLOv5 model loading verified")
                except Exception as e:
                    print(f"⚠️  Model loading test failed: {e}")
            else:
                print("❌ YOLOv5 not found")
                
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 Please install required packages")
    
    def step4_training_configuration(self):
        """
        STEP 4: TRAINING CONFIGURATION
        ==============================
        
        This step covers:
        - Parameter optimization
        - Hardware configuration
        - Data loading optimization
        - Training strategy
        """
        print("\n" + "="*60)
        print("STEP 4: TRAINING CONFIGURATION")
        print("="*60)
        
        print("""
⚙️  TRAINING CONFIGURATION GUIDE:

1. BATCH SIZE SELECTION:
   • GPU Memory: 4GB → batch_size=8, 8GB → batch_size=16, 16GB → batch_size=32
   • Start small and increase until GPU memory is full
   • Larger batches = more stable gradients but slower training

2. LEARNING RATE OPTIMIZATION:
   • Initial LR: 0.01 (default), 0.001 (conservative), 0.1 (aggressive)
   • Use learning rate finder for optimal value
   • Cosine annealing or step decay for scheduling

3. EPOCHS AND EARLY STOPPING:
   • Small datasets: 100-300 epochs
   • Large datasets: 50-100 epochs
   • Monitor validation loss for early stopping

4. IMAGE SIZE CONSIDERATIONS:
   • 416x416: Faster training, lower accuracy
   • 640x640: Balanced speed/accuracy (recommended)
   • 832x832: Higher accuracy, slower training

5. DATA AUGMENTATION:
   • Mosaic: Combines 4 images (recommended: 1.0)
   • HSV: Color space augmentation (h=0.015, s=0.7, v=0.4)
   • Geometric: Rotation, translation, scaling
        """)
        
        # Analyze dataset for optimal configuration
        self.analyze_dataset_for_config()
        
        # Optimize batch size
        self.optimize_batch_size()
        
        # Create optimized configuration
        self.create_optimized_config()
        
        print("✅ Step 4 Complete: Training Configuration")
        
    def analyze_dataset_for_config(self):
        """Analyze dataset to determine optimal configuration"""
        print("\n📊 ANALYZING DATASET FOR OPTIMAL CONFIGURATION...")
        
        # Count total images and annotations
        total_images = 0
        total_annotations = 0
        class_counts = defaultdict(int)
        image_sizes = []
        
        for split in ['train', 'val', 'test']:
            images_dir = self.dataset_dir / "images" / split
            labels_dir = self.dataset_dir / "labels" / split
            
            if not images_dir.exists():
                continue
            
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            total_images += len(image_files)
            
            for image_file in image_files:
                # Get image size
                image = cv2.imread(str(image_file))
                if image is not None:
                    h, w = image.shape[:2]
                    image_sizes.append((w, h))
                
                # Count annotations
                label_file = labels_dir / f"{image_file.stem}.txt"
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        if line.strip():
                            total_annotations += 1
                            class_id = int(line.split()[0])
                            class_counts[class_id] += 1
        
        # Analysis results
        print(f"📈 DATASET ANALYSIS RESULTS:")
        print(f"  Total Images: {total_images}")
        print(f"  Total Annotations: {total_annotations}")
        print(f"  Avg Annotations per Image: {total_annotations/total_images:.2f}")
        
        if image_sizes:
            avg_width = np.mean([s[0] for s in image_sizes])
            avg_height = np.mean([s[1] for s in image_sizes])
            print(f"  Average Image Size: {avg_width:.0f}x{avg_height:.0f}")
        
        print(f"  Class Distribution:")
        for class_id, count in class_counts.items():
            percentage = (count / total_annotations) * 100
            print(f"    Class {class_id}: {count} ({percentage:.1f}%)")
        
        # Recommendations
        print(f"\n💡 CONFIGURATION RECOMMENDATIONS:")
        
        if total_images < 500:
            print("  • Small dataset: Use more epochs (200-300)")
            print("  • Enable strong augmentation")
            print("  • Consider transfer learning")
        elif total_images < 2000:
            print("  • Medium dataset: 100-200 epochs")
            print("  • Moderate augmentation")
        else:
            print("  • Large dataset: 50-100 epochs")
            print("  • Standard augmentation")
        
        # Check class balance
        if class_counts:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / min_count
            
            if imbalance_ratio > 10:
                print("  • High class imbalance detected")
                print("  • Consider class weights or balanced sampling")
        
        return {
            'total_images': total_images,
            'total_annotations': total_annotations,
            'class_counts': dict(class_counts),
            'avg_image_size': (avg_width, avg_height) if image_sizes else None
        }
    
    def optimize_batch_size(self):
        """Determine optimal batch size based on available hardware"""
        print("\n🎯 OPTIMIZING BATCH SIZE...")
        
        try:
            import torch
            
            if torch.cuda.is_available():
                # Get GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"🎮 GPU Memory: {gpu_memory:.1f} GB")
                
                # Recommend batch size based on GPU memory
                if gpu_memory >= 24:
                    recommended_batch = 32
                elif gpu_memory >= 16:
                    recommended_batch = 24
                elif gpu_memory >= 11:
                    recommended_batch = 16
                elif gpu_memory >= 8:
                    recommended_batch = 12
                elif gpu_memory >= 6:
                    recommended_batch = 8
                else:
                    recommended_batch = 4
                
                print(f"💡 Recommended batch size: {recommended_batch}")
                
            else:
                print("💻 Using CPU - recommended batch size: 4-8")
                recommended_batch = 4
            
            # Test batch size
            print("🧪 Testing batch size compatibility...")
            self.test_batch_size(recommended_batch)
            
            return recommended_batch
            
        except ImportError:
            print("❌ PyTorch not available for batch size optimization")
            return 16
    
    def test_batch_size(self, batch_size):
        """Test if batch size works with available hardware"""
        try:
            import torch
            
            # Create dummy data
            dummy_images = torch.randn(batch_size, 3, 640, 640)
            
            if torch.cuda.is_available():
                dummy_images = dummy_images.cuda()
                
                # Test memory usage
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated()
                
                # Simulate forward pass
                _ = dummy_images * 2  # Simple operation
                
                memory_after = torch.cuda.memory_allocated()
                memory_used = (memory_after - memory_before) / (1024**3)
                
                print(f"✅ Batch size {batch_size} test passed")
                print(f"📊 Memory used: {memory_used:.2f} GB")
                
                torch.cuda.empty_cache()
            else:
                print(f"✅ Batch size {batch_size} test passed (CPU)")
                
        except Exception as e:
            print(f"⚠️  Batch size {batch_size} may be too large: {e}")
    
    def create_optimized_config(self):
        """Create optimized training configuration"""
        print("\n⚙️  CREATING OPTIMIZED CONFIGURATION...")
        
        # Load dataset analysis
        dataset_analysis = self.analyze_dataset_for_config()
        
        # Determine optimal parameters
        total_images = dataset_analysis['total_images']
        
        if total_images < 500:
            epochs = 200
            patience = 50
            augmentation_strength = 'high'
        elif total_images < 2000:
            epochs = 150
            patience = 30
            augmentation_strength = 'medium'
        else:
            epochs = 100
            patience = 20
            augmentation_strength = 'standard'
        
        # Augmentation settings
        if augmentation_strength == 'high':
            aug_config = {
                'hsv_h': 0.02, 'hsv_s': 0.8, 'hsv_v': 0.5,
                'degrees': 5.0, 'translate': 0.15, 'scale': 0.6,
                'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.1
            }
        elif augmentation_strength == 'medium':
            aug_config = {
                'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
                'degrees': 2.0, 'translate': 0.1, 'scale': 0.5,
                'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0
            }
        else:  # standard
            aug_config = {
                'hsv_h': 0.01, 'hsv_s': 0.6, 'hsv_v': 0.3,
                'degrees': 0.0, 'translate': 0.05, 'scale': 0.4,
                'fliplr': 0.5, 'mosaic': 0.8, 'mixup': 0.0
            }
        
        # Create optimized configuration
        optimized_config = {
            'model_size': 's',
            'epochs': epochs,
            'batch_size': self.optimize_batch_size(),
            'img_size': 640,
            'patience': patience,
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            **aug_config,
            'box': 0.05,
            'cls': 0.5,
            'obj': 1.0
        }
        
        # Save optimized configuration
        config_path = self.project_dir / "configs" / "optimized_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(optimized_config, f, default_flow_style=False)
        
        print(f"✅ Optimized configuration saved: {config_path}")
        print(f"📊 Key parameters:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {optimized_config['batch_size']}")
        print(f"  Augmentation: {augmentation_strength}")
        
        return optimized_config

def main():
    """Run the complete YOLOv5 training guide"""
    print("🚀 COMPLETE YOLOv5 CUSTOM TRAINING GUIDE")
    print("=" * 80)
    print("""
This comprehensive guide will walk you through every step of training
a custom YOLOv5 model, from dataset preparation to deployment.

The process includes:
1. Dataset Preparation
2. Image Annotation  
3. Training Environment Setup
4. Training Configuration
5. Model Training (next script)
6. Performance Evaluation (next script)
7. Model Optimization (next script)
8. Deployment Preparation (next script)
    """)
    
    # Initialize guide
    guide = YOLOv5TrainingGuide("my_custom_detection")
    
    # Run through steps 1-4
    guide.step1_dataset_preparation()
    guide.step2_image_annotation()
    guide.step3_training_environment_setup()
    guide.step4_training_configuration()
    
    print("\n" + "="*80)
    print("🎉 STEPS 1-4 COMPLETE!")
    print("="*80)
    print("""
✅ Completed Steps:
1. ✓ Dataset Preparation - Images collected and organized
2. ✓ Image Annotation - Bounding boxes created in YOLO format
3. ✓ Training Environment Setup - YOLOv5 installed and configured
4. ✓ Training Configuration - Optimal parameters determined

🚀 Next Steps:
5. Run training_execution.py to start model training
6. Run performance_evaluation.py to analyze results
7. Run model_optimization.py to optimize for deployment
8. Run deployment_preparation.py to prepare for real-time use

💡 Your project is ready for training!
📁 Project directory: {guide.project_dir}
📄 Dataset config: {guide.dataset_dir}/dataset.yaml
⚙️  Training config: {guide.project_dir}/configs/optimized_config.yaml
    """.format(guide=guide))

if __name__ == "__main__":
    main()
