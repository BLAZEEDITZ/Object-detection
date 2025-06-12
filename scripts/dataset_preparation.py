import os
import shutil
import json
import cv2
import numpy as np
from pathlib import Path
import requests
from PIL import Image
import yaml
from collections import defaultdict
import random

class DatasetManager:
    def __init__(self, dataset_name="custom_dataset"):
        """
        Initialize dataset manager for YOLOv5 custom training
        
        Args:
            dataset_name: Name of the dataset
        """
        self.dataset_name = dataset_name
        self.dataset_path = Path(f"datasets/{dataset_name}")
        self.classes = []
        self.class_to_id = {}
        
        # Create dataset structure
        self.setup_dataset_structure()
        
        print(f"🗂️  Dataset Manager initialized: {dataset_name}")
        print(f"📁 Dataset path: {self.dataset_path}")
    
    def setup_dataset_structure(self):
        """Create YOLOv5 dataset directory structure"""
        directories = [
            "images/train",
            "images/val", 
            "images/test",
            "labels/train",
            "labels/val",
            "labels/test",
            "raw_images",
            "annotations"
        ]
        
        for directory in directories:
            (self.dataset_path / directory).mkdir(parents=True, exist_ok=True)
        
        print("✅ Dataset directory structure created")
    
    def add_class(self, class_name):
        """
        Add a new class to the dataset
        
        Args:
            class_name: Name of the object class
        """
        if class_name not in self.classes:
            self.classes.append(class_name)
            self.class_to_id[class_name] = len(self.classes) - 1
            print(f"➕ Added class: {class_name} (ID: {self.class_to_id[class_name]})")
        else:
            print(f"⚠️  Class '{class_name}' already exists")
    
    def create_dataset_yaml(self):
        """Create dataset.yaml file for YOLOv5 training"""
        dataset_config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        yaml_path = self.dataset_path / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"📄 Dataset YAML created: {yaml_path}")
        return yaml_path
    
    def collect_images_from_web(self, search_terms, images_per_term=50):
        """
        Collect images from web for dataset creation
        
        Args:
            search_terms: List of search terms for image collection
            images_per_term: Number of images to collect per term
        """
        print(f"🌐 Collecting images from web...")
        
        # This is a placeholder - in real implementation, you would use:
        # - Bing Image Search API
        # - Google Custom Search API
        # - Web scraping tools
        # - Or manual collection
        
        collected_images = []
        
        for term in search_terms:
            print(f"🔍 Searching for: {term}")
            
            # Simulate image collection
            for i in range(min(images_per_term, 10)):  # Limited for demo
                # Create synthetic images for demonstration
                image = self.create_synthetic_image(term, i)
                
                # Save image
                image_name = f"{term}_{i:03d}.jpg"
                image_path = self.dataset_path / "raw_images" / image_name
                cv2.imwrite(str(image_path), image)
                
                collected_images.append({
                    'path': image_path,
                    'term': term,
                    'filename': image_name
                })
        
        print(f"✅ Collected {len(collected_images)} images")
        return collected_images
    
    def create_synthetic_image(self, object_type, index):
        """
        Create synthetic training images for demonstration
        
        Args:
            object_type: Type of object to create
            index: Image index
        """
        # Create base image
        image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        
        # Add synthetic objects based on type
        if "car" in object_type.lower():
            # Draw car-like rectangle
            cv2.rectangle(image, (100 + index*10, 200), (300 + index*10, 350), (0, 0, 255), -1)
            cv2.rectangle(image, (120 + index*10, 220), (280 + index*10, 280), (100, 100, 100), -1)
        elif "person" in object_type.lower():
            # Draw person-like shape
            cv2.rectangle(image, (200 + index*5, 100), (250 + index*5, 300), (255, 200, 100), -1)
            cv2.circle(image, (225 + index*5, 80), 25, (255, 200, 100), -1)
        elif "phone" in object_type.lower():
            # Draw phone-like rectangle
            cv2.rectangle(image, (250 + index*3, 150), (300 + index*3, 250), (50, 50, 50), -1)
        else:
            # Generic object
            cv2.rectangle(image, (150 + index*8, 150), (250 + index*8, 250), (100, 255, 100), -1)
        
        return image
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """
        Split dataset into train/validation/test sets
        
        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data  
            test_ratio: Ratio of test data
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Get all images
        raw_images_path = self.dataset_path / "raw_images"
        image_files = list(raw_images_path.glob("*.jpg")) + list(raw_images_path.glob("*.png"))
        
        if not image_files:
            print("❌ No images found in raw_images directory")
            return
        
        # Shuffle images
        random.shuffle(image_files)
        
        # Calculate split indices
        total_images = len(image_files)
        train_end = int(total_images * train_ratio)
        val_end = train_end + int(total_images * val_ratio)
        
        # Split files
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        # Copy files to respective directories
        splits = [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files)
        ]
        
        for split_name, files in splits:
            print(f"📂 Processing {split_name} split: {len(files)} images")
            
            for file_path in files:
                # Copy image
                dest_image = self.dataset_path / "images" / split_name / file_path.name
                shutil.copy2(file_path, dest_image)
        
        print(f"✅ Dataset split complete:")
        print(f"  📊 Train: {len(train_files)} images")
        print(f"  📊 Val: {len(val_files)} images") 
        print(f"  📊 Test: {len(test_files)} images")
        
        return {
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files)
        }
    
    def generate_sample_annotations(self):
        """
        Generate sample annotations for demonstration
        In real use, you would use annotation tools like LabelImg
        """
        print("🏷️  Generating sample annotations...")
        
        splits = ['train', 'val', 'test']
        
        for split in splits:
            images_dir = self.dataset_path / "images" / split
            labels_dir = self.dataset_path / "labels" / split
            
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            
            for image_file in image_files:
                # Load image to get dimensions
                image = cv2.imread(str(image_file))
                if image is None:
                    continue
                    
                height, width = image.shape[:2]
                
                # Generate sample annotation
                annotation_lines = []
                
                # Add 1-3 random bounding boxes
                num_objects = random.randint(1, 3)
                
                for _ in range(num_objects):
                    # Random class
                    class_id = random.randint(0, max(0, len(self.classes) - 1))
                    
                    # Random bounding box (normalized coordinates)
                    x_center = random.uniform(0.2, 0.8)
                    y_center = random.uniform(0.2, 0.8)
                    box_width = random.uniform(0.1, 0.3)
                    box_height = random.uniform(0.1, 0.3)
                    
                    # Ensure box stays within image
                    x_center = max(box_width/2, min(1 - box_width/2, x_center))
                    y_center = max(box_height/2, min(1 - box_height/2, y_center))
                    
                    annotation_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
                
                # Save annotation file
                label_file = labels_dir / f"{image_file.stem}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(annotation_lines))
        
        print("✅ Sample annotations generated")
    
    def validate_dataset(self):
        """Validate dataset structure and annotations"""
        print("🔍 Validating dataset...")
        
        validation_results = {
            'total_images': 0,
            'total_labels': 0,
            'missing_labels': [],
            'invalid_annotations': [],
            'class_distribution': defaultdict(int)
        }
        
        splits = ['train', 'val', 'test']
        
        for split in splits:
            images_dir = self.dataset_path / "images" / split
            labels_dir = self.dataset_path / "labels" / split
            
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            validation_results['total_images'] += len(image_files)
            
            for image_file in image_files:
                label_file = labels_dir / f"{image_file.stem}.txt"
                
                if not label_file.exists():
                    validation_results['missing_labels'].append(str(image_file))
                    continue
                
                validation_results['total_labels'] += 1
                
                # Validate annotation format
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            validation_results['invalid_annotations'].append(
                                f"{label_file}:{line_num} - Invalid format"
                            )
                            continue
                        
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        
                        # Check coordinate ranges
                        if not all(0 <= coord <= 1 for coord in coords):
                            validation_results['invalid_annotations'].append(
                                f"{label_file}:{line_num} - Coordinates out of range"
                            )
                        
                        # Update class distribution
                        if 0 <= class_id < len(self.classes):
                            validation_results['class_distribution'][self.classes[class_id]] += 1
                
                except Exception as e:
                    validation_results['invalid_annotations'].append(
                        f"{label_file} - Error: {str(e)}"
                    )
        
        # Print validation results
        print(f"📊 Validation Results:")
        print(f"  Total Images: {validation_results['total_images']}")
        print(f"  Total Labels: {validation_results['total_labels']}")
        print(f"  Missing Labels: {len(validation_results['missing_labels'])}")
        print(f"  Invalid Annotations: {len(validation_results['invalid_annotations'])}")
        
        if validation_results['class_distribution']:
            print(f"  Class Distribution:")
            for class_name, count in validation_results['class_distribution'].items():
                print(f"    {class_name}: {count}")
        
        # Show issues if any
        if validation_results['missing_labels']:
            print(f"\n⚠️  Missing labels (first 5):")
            for missing in validation_results['missing_labels'][:5]:
                print(f"    {missing}")
        
        if validation_results['invalid_annotations']:
            print(f"\n⚠️  Invalid annotations (first 5):")
            for invalid in validation_results['invalid_annotations'][:5]:
                print(f"    {invalid}")
        
        return validation_results

def main():
    """Demonstrate dataset preparation"""
    print("🚀 YOLOv5 Custom Dataset Preparation")
    print("=" * 60)
    
    # Initialize dataset manager
    dataset_manager = DatasetManager("custom_objects")
    
    # Add custom classes
    print("\n📋 Adding custom object classes...")
    custom_classes = ["custom_car", "custom_person", "custom_phone", "custom_laptop"]
    
    for class_name in custom_classes:
        dataset_manager.add_class(class_name)
    
    # Collect sample images
    print("\n🖼️  Collecting sample images...")
    search_terms = ["car", "person", "phone", "laptop"]
    collected_images = dataset_manager.collect_images_from_web(search_terms, images_per_term=20)
    
    # Split dataset
    print("\n📂 Splitting dataset...")
    split_info = dataset_manager.split_dataset(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    
    # Generate sample annotations
    print("\n🏷️  Generating annotations...")
    dataset_manager.generate_sample_annotations()
    
    # Create dataset YAML
    print("\n📄 Creating dataset configuration...")
    yaml_path = dataset_manager.create_dataset_yaml()
    
    # Validate dataset
    print("\n🔍 Validating dataset...")
    validation_results = dataset_manager.validate_dataset()
    
    print("\n✅ Dataset preparation complete!")
    print(f"📁 Dataset location: {dataset_manager.dataset_path}")
    print(f"📄 Configuration file: {yaml_path}")
    print("\n💡 Next steps:")
    print("1. Review and correct annotations using LabelImg or similar tool")
    print("2. Add more real images to improve dataset quality")
    print("3. Run the training script with this dataset")

if __name__ == "__main__":
    main()
