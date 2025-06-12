import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

class AdvancedYOLOv5Detector:
    def __init__(self, model_name='yolov5s'):
        """Advanced YOLOv5 detector with additional features"""
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.classes = self.model.names
        self.detection_history = []
        self.class_colors = self.generate_colors()
        
    def generate_colors(self):
        """Generate unique colors for each class"""
        colors = {}
        np.random.seed(42)  # For consistent colors
        for class_id, class_name in self.classes.items():
            colors[class_name] = tuple(map(int, np.random.randint(0, 255, 3)))
        return colors
    
    def detect_with_tracking(self, image, confidence_threshold=0.25, iou_threshold=0.45):
        """
        Advanced detection with confidence and IoU thresholds
        
        Args:
            image: Input image
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model.conf = confidence_threshold
        self.model.iou = iou_threshold
        
        results = self.model(image)
        
        # Store detection history
        detection_info = {
            'timestamp': time.time(),
            'detections': results.pandas().xyxy[0] if len(results.pandas().xyxy[0]) > 0 else None,
            'confidence_threshold': confidence_threshold,
            'iou_threshold': iou_threshold
        }
        self.detection_history.append(detection_info)
        
        return results
    
    def draw_advanced_detections(self, image, results, show_confidence=True, show_class_colors=True):
        """
        Draw detections with advanced visualization options
        """
        if results is None:
            return image
            
        detections = results.pandas().xyxy[0]
        annotated_image = image.copy()
        
        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            confidence = detection['confidence']
            class_name = detection['name']
            
            # Get color for this class
            if show_class_colors and class_name in self.class_colors:
                color = self.class_colors[class_name]
            else:
                color = (0, 255, 0)  # Default green
            
            # Draw bounding box with varying thickness based on confidence
            thickness = max(1, int(confidence * 4))
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            if show_confidence:
                label = f"{class_name}: {confidence:.2f}"
            else:
                label = class_name
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        return annotated_image
    
    def analyze_detections(self, results):
        """
        Analyze detection results and provide statistics
        """
        if results is None:
            return {}
            
        detections = results.pandas().xyxy[0]
        
        analysis = {
            'total_objects': len(detections),
            'classes_detected': detections['name'].unique().tolist(),
            'class_counts': detections['name'].value_counts().to_dict(),
            'confidence_stats': {
                'mean': detections['confidence'].mean(),
                'min': detections['confidence'].min(),
                'max': detections['confidence'].max(),
                'std': detections['confidence'].std()
            },
            'bounding_box_areas': []
        }
        
        # Calculate bounding box areas
        for _, detection in detections.iterrows():
            width = detection['xmax'] - detection['xmin']
            height = detection['ymax'] - detection['ymin']
            area = width * height
            analysis['bounding_box_areas'].append(area)
        
        if analysis['bounding_box_areas']:
            analysis['area_stats'] = {
                'mean': np.mean(analysis['bounding_box_areas']),
                'min': np.min(analysis['bounding_box_areas']),
                'max': np.max(analysis['bounding_box_areas']),
                'std': np.std(analysis['bounding_box_areas'])
            }
        
        return analysis
    
    def create_detection_report(self, image_path, confidence_threshold=0.25):
        """
        Create a comprehensive detection report
        """
        print(f"\n📊 Detection Report for: {image_path}")
        print("=" * 60)
        
        # Load and process image
        if isinstance(image_path, str) and image_path.startswith('http'):
            import requests
            from PIL import Image
            from io import BytesIO
            response = requests.get(image_path)
            image = np.array(Image.open(BytesIO(response.content)))
        else:
            image = cv2.imread(image_path) if isinstance(image_path, str) else image_path
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect objects
        results = self.detect_with_tracking(image, confidence_threshold)
        analysis = self.analyze_detections(results)
        
        # Print report
        print(f"🎯 Total Objects Detected: {analysis['total_objects']}")
        print(f"📋 Classes Found: {', '.join(analysis['classes_detected'])}")
        
        if analysis['class_counts']:
            print("\n📈 Object Counts by Class:")
            for class_name, count in analysis['class_counts'].items():
                print(f"  {class_name}: {count}")
        
        if 'confidence_stats' in analysis:
            conf_stats = analysis['confidence_stats']
            print(f"\n🎲 Confidence Statistics:")
            print(f"  Mean: {conf_stats['mean']:.3f}")
            print(f"  Range: {conf_stats['min']:.3f} - {conf_stats['max']:.3f}")
            print(f"  Std Dev: {conf_stats['std']:.3f}")
        
        # Create visualization
        annotated_image = self.draw_advanced_detections(image, results)
        
        # Display results
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Annotated image
        axes[1].imshow(annotated_image)
        axes[1].set_title(f'Detected Objects (Confidence ≥ {confidence_threshold})')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return analysis, annotated_image
    
    def compare_thresholds(self, image, thresholds=[0.1, 0.25, 0.5, 0.75]):
        """
        Compare detection results across different confidence thresholds
        """
        print(f"\n🔍 Comparing Detection Thresholds")
        print("=" * 50)
        
        results_comparison = {}
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, threshold in enumerate(thresholds):
            if i >= 4:  # Limit to 4 subplots
                break
                
            results = self.detect_with_tracking(image, threshold)
            analysis = self.analyze_detections(results)
            results_comparison[threshold] = analysis
            
            annotated = self.draw_advanced_detections(image, results)
            
            axes[i].imshow(annotated)
            axes[i].set_title(f'Confidence ≥ {threshold} ({analysis["total_objects"]} objects)')
            axes[i].axis('off')
            
            print(f"Threshold {threshold}: {analysis['total_objects']} objects detected")
        
        plt.tight_layout()
        plt.show()
        
        return results_comparison

def main():
    """Demonstrate advanced YOLOv5 features"""
    print("🚀 Advanced YOLOv5 Object Detection")
    print("=" * 50)
    
    # Initialize advanced detector
    detector = AdvancedYOLOv5Detector('yolov5s')
    
    # Example 1: Comprehensive detection report
    print("\n📊 Example 1: Comprehensive Detection Report")
    sample_url = "https://ultralytics.com/images/bus.jpg"
    
    try:
        analysis, annotated = detector.create_detection_report(sample_url, confidence_threshold=0.25)
    except Exception as e:
        print(f"❌ Error in detection report: {e}")
    
    # Example 2: Threshold comparison
    print("\n🔍 Example 2: Threshold Comparison")
    
    # Create a test image with multiple objects
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 240
    
    # Add various shapes and patterns
    cv2.rectangle(test_image, (50, 50), (150, 150), (255, 100, 100), -1)
    cv2.rectangle(test_image, (200, 200), (350, 350), (100, 255, 100), -1)
    cv2.circle(test_image, (500, 100), 60, (100, 100, 255), -1)
    cv2.ellipse(test_image, (400, 300), (80, 40), 45, 0, 360, (255, 255, 100), -1)
    
    try:
        comparison_results = detector.compare_thresholds(
            test_image, 
            thresholds=[0.1, 0.25, 0.5, 0.75]
        )
        
        print("\n📈 Threshold Comparison Summary:")
        for threshold, analysis in comparison_results.items():
            print(f"  Threshold {threshold}: {analysis['total_objects']} objects")
            
    except Exception as e:
        print(f"❌ Error in threshold comparison: {e}")
    
    # Example 3: Show class colors
    print(f"\n🎨 Example 3: Class Color Mapping (showing first 10)")
    for i, (class_name, color) in enumerate(list(detector.class_colors.items())[:10]):
        print(f"  {class_name}: RGB{color}")
    
    print("\n✅ Advanced detection demonstration complete!")
    print("\n💡 Advanced Features:")
    print("  - Confidence and IoU threshold tuning")
    print("  - Class-specific color coding")
    print("  - Detection statistics and analysis")
    print("  - Threshold comparison visualization")
    print("  - Detection history tracking")

if __name__ == "__main__":
    main()
