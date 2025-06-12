import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

class YOLOv5ObjectDetector:
    def __init__(self, model_name='yolov5s', confidence_threshold=0.25):
        """
        Initialize YOLOv5 Object Detector
        
        Args:
            model_name (str): YOLOv5 model variant ('yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
            confidence_threshold (float): Confidence threshold for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        
    def load_model(self, model_name):
        """Load YOLOv5 model"""
        try:
            # Load model from torch hub
            model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            model.conf = self.confidence_threshold
            print(f"✅ Successfully loaded {model_name}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def detect_objects(self, image_path_or_array):
        """
        Detect objects in an image
        
        Args:
            image_path_or_array: Path to image file or numpy array
            
        Returns:
            results: YOLOv5 detection results
        """
        if self.model is None:
            print("❌ Model not loaded")
            return None
            
        try:
            results = self.model(image_path_or_array)
            return results
        except Exception as e:
            print(f"❌ Error during detection: {e}")
            return None
    
    def draw_detections(self, image, results):
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Input image (numpy array)
            results: YOLOv5 detection results
            
        Returns:
            annotated_image: Image with drawn detections
        """
        if results is None:
            return image
            
        # Get detections
        detections = results.pandas().xyxy[0]
        
        # Create a copy of the image
        annotated_image = image.copy()
        
        # Draw each detection
        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            confidence = detection['confidence']
            class_name = detection['name']
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_image
    
    def process_image(self, image_path, save_path=None, show_result=True):
        """
        Process a single image for object detection
        
        Args:
            image_path (str): Path to input image
            save_path (str): Path to save annotated image (optional)
            show_result (bool): Whether to display the result
        """
        try:
            # Load image
            if isinstance(image_path, str):
                if image_path.startswith('http'):
                    # Download image from URL
                    response = requests.get(image_path)
                    image = np.array(Image.open(BytesIO(response.content)))
                else:
                    # Load local image
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = image_path
            
            print(f"📸 Processing image...")
            
            # Detect objects
            results = self.detect_objects(image)
            
            if results is not None:
                # Print detection summary
                detections = results.pandas().xyxy[0]
                print(f"🎯 Found {len(detections)} objects:")
                
                for _, detection in detections.iterrows():
                    class_name = detection['name']
                    confidence = detection['confidence']
                    print(f"  - {class_name}: {confidence:.2f}")
                
                # Draw detections
                annotated_image = self.draw_detections(image, results)
                
                # Save result if path provided
                if save_path:
                    plt.figure(figsize=(12, 8))
                    plt.imshow(annotated_image)
                    plt.axis('off')
                    plt.title('Object Detection Results')
                    plt.savefig(save_path, bbox_inches='tight', dpi=150)
                    print(f"💾 Saved result to {save_path}")
                
                # Show result
                if show_result:
                    plt.figure(figsize=(12, 8))
                    plt.imshow(annotated_image)
                    plt.axis('off')
                    plt.title('Object Detection Results')
                    plt.show()
                
                return annotated_image, results
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return None, None
    
    def process_video(self, video_path, output_path=None, max_frames=None):
        """
        Process video for object detection
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video (optional)
            max_frames (int): Maximum number of frames to process (optional)
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"❌ Error opening video: {video_path}")
                return
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"🎬 Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
            
            # Setup video writer if output path provided
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames if max_frames specified
                if max_frames and processed_frames >= max_frames:
                    break
                
                # Convert BGR to RGB for YOLOv5
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect objects
                results = self.detect_objects(rgb_frame)
                
                if results is not None:
                    # Draw detections
                    annotated_frame = self.draw_detections(rgb_frame, results)
                    # Convert back to BGR for video writer
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                else:
                    annotated_frame = frame
                
                # Write frame if output specified
                if output_path:
                    out.write(annotated_frame)
                
                processed_frames += 1
                
                # Print progress
                if processed_frames % 30 == 0:
                    print(f"📹 Processed {processed_frames}/{min(max_frames or total_frames, total_frames)} frames")
            
            # Cleanup
            cap.release()
            if output_path:
                out.release()
                print(f"💾 Saved processed video to {output_path}")
            
            print(f"✅ Video processing complete! Processed {processed_frames} frames")
            
        except Exception as e:
            print(f"❌ Error processing video: {e}")

def main():
    """Main function to demonstrate object detection"""
    print("🚀 YOLOv5 Object Detection Program")
    print("=" * 50)
    
    # Initialize detector
    detector = YOLOv5ObjectDetector(model_name='yolov5s', confidence_threshold=0.25)
    
    # Example 1: Detect objects in a sample image from URL
    print("\n📸 Example 1: Processing sample image from URL")
    sample_image_url = "https://ultralytics.com/images/bus.jpg"
    
    try:
        annotated_image, results = detector.process_image(
            sample_image_url, 
            save_path="detection_result.jpg",
            show_result=True
        )
    except Exception as e:
        print(f"❌ Error with sample image: {e}")
    
    # Example 2: Create a simple test image with objects
    print("\n🎨 Example 2: Creating and processing a test image")
    
    # Create a simple test image with shapes (simulating objects)
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Add some colored rectangles (simulating objects)
    cv2.rectangle(test_image, (50, 50), (200, 150), (255, 0, 0), -1)  # Red rectangle
    cv2.rectangle(test_image, (300, 200), (500, 350), (0, 255, 0), -1)  # Green rectangle
    cv2.circle(test_image, (400, 100), 50, (0, 0, 255), -1)  # Blue circle
    
    # Add some text
    cv2.putText(test_image, "Test Image", (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    try:
        annotated_test, test_results = detector.process_image(
            test_image,
            save_path="test_detection_result.jpg",
            show_result=True
        )
    except Exception as e:
        print(f"❌ Error with test image: {e}")
    
    # Example 3: Show available classes
    print(f"\n📋 Available object classes ({len(detector.classes)} total):")
    for i, class_name in enumerate(detector.classes.values()):
        if i < 10:  # Show first 10 classes
            print(f"  {i}: {class_name}")
        elif i == 10:
            print(f"  ... and {len(detector.classes) - 10} more classes")
            break
    
    print("\n✅ Object detection demonstration complete!")
    print("\n💡 Tips:")
    print("  - Use 'yolov5s' for speed, 'yolov5x' for accuracy")
    print("  - Adjust confidence_threshold (0.1-0.9) based on your needs")
    print("  - For video processing, consider using max_frames to limit processing time")

if __name__ == "__main__":
    main()
