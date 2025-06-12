import cv2
import torch
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt

class RealTimeYOLOv5:
    def __init__(self, model_name='yolov5s', confidence_threshold=0.25):
        """
        Real-time YOLOv5 object detection
        
        Args:
            model_name: YOLOv5 model variant
            confidence_threshold: Detection confidence threshold
        """
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.conf = confidence_threshold
        self.classes = self.model.names
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.detection_counts = deque(maxlen=100)
        
        # Colors for different classes
        np.random.seed(42)
        self.colors = {}
        for class_id, class_name in self.classes.items():
            self.colors[class_name] = tuple(map(int, np.random.randint(0, 255, 3)))
    
    def process_frame(self, frame):
        """
        Process a single frame for object detection
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            annotated_frame: Frame with detections drawn
            detections: Detection results
            fps: Current FPS
        """
        start_time = time.time()
        
        # Convert BGR to RGB for YOLOv5
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = self.model(rgb_frame)
        detections = results.pandas().xyxy[0]
        
        # Draw detections
        annotated_frame = self.draw_detections(frame, detections)
        
        # Calculate FPS
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        self.fps_history.append(fps)
        self.detection_counts.append(len(detections))
        
        # Add FPS and detection count to frame
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Objects: {len(detections)}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame, detections, fps
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        annotated_frame = frame.copy()
        
        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            confidence = detection['confidence']
            class_name = detection['name']
            
            # Get color for this class
            color = self.colors.get(class_name, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame
    
    def webcam_detection(self, camera_index=0, display_size=(640, 480)):
        """
        Run real-time detection on webcam feed
        
        Args:
            camera_index: Camera device index (usually 0 for default camera)
            display_size: Size of display window
        """
        print(f"🎥 Starting webcam detection (Camera {camera_index})")
        print("Press 'q' to quit, 's' to save screenshot")
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_index}")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_size[1])
        
        screenshot_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error reading from camera")
                    break
                
                # Process frame
                annotated_frame, detections, fps = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('YOLOv5 Real-time Detection', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_count += 1
                    filename = f"detection_screenshot_{screenshot_count}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"📸 Screenshot saved: {filename}")
        
        except KeyboardInterrupt:
            print("\n⏹️ Detection stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print performance summary
            if self.fps_history:
                avg_fps = np.mean(self.fps_history)
                avg_detections = np.mean(self.detection_counts)
                print(f"\n📊 Performance Summary:")
                print(f"  Average FPS: {avg_fps:.1f}")
                print(f"  Average Objects per Frame: {avg_detections:.1f}")
    
    def video_file_detection(self, video_path, output_path=None, max_frames=None):
        """
        Process video file for object detection
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            max_frames: Maximum frames to process (optional)
        """
        print(f"🎬 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_summary = {}
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if max_frames and frame_count > max_frames:
                    break
                
                # Process frame
                annotated_frame, detections, current_fps = self.process_frame(frame)
                
                # Update detection summary
                for _, detection in detections.iterrows():
                    class_name = detection['name']
                    detection_summary[class_name] = detection_summary.get(class_name, 0) + 1
                
                # Write frame
                if output_path:
                    out.write(annotated_frame)
                
                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / min(max_frames or total_frames, total_frames)) * 100
                    print(f"📈 Progress: {progress:.1f}% ({frame_count} frames)")
        
        except KeyboardInterrupt:
            print("\n⏹️ Processing stopped by user")
        
        finally:
            cap.release()
            if output_path:
                out.release()
                print(f"💾 Output saved: {output_path}")
            
            # Print detection summary
            print(f"\n📊 Detection Summary ({frame_count} frames processed):")
            for class_name, count in sorted(detection_summary.items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {count} detections")
    
    def create_performance_plot(self):
        """Create performance visualization"""
        if not self.fps_history or not self.detection_counts:
            print("❌ No performance data available")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # FPS plot
        ax1.plot(list(self.fps_history), 'b-', linewidth=2)
        ax1.set_title('Real-time FPS Performance')
        ax1.set_ylabel('FPS')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=np.mean(self.fps_history), color='r', linestyle='--', 
                   label=f'Average: {np.mean(self.fps_history):.1f} FPS')
        ax1.legend()
        
        # Detection count plot
        ax2.plot(list(self.detection_counts), 'g-', linewidth=2)
        ax2.set_title('Objects Detected per Frame')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Object Count')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=np.mean(self.detection_counts), color='r', linestyle='--',
                   label=f'Average: {np.mean(self.detection_counts):.1f} objects')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    """Demonstrate real-time YOLOv5 detection"""
    print("🚀 Real-time YOLOv5 Object Detection")
    print("=" * 50)
    
    # Initialize real-time detector
    detector = RealTimeYOLOv5('yolov5s', confidence_threshold=0.25)
    
    print("\n🎥 Real-time Detection Options:")
    print("1. Webcam detection (requires camera)")
    print("2. Video file processing")
    print("3. Performance analysis")
    
    # For demonstration, we'll simulate video processing
    print("\n📹 Simulating video processing...")
    
    # Create synthetic video frames for demonstration
    print("🎬 Creating synthetic video frames for demo...")
    
    # Simulate processing multiple frames
    synthetic_frames = []
    for i in range(10):
        # Create a frame with moving objects
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some geometric shapes that might be detected
        cv2.rectangle(frame, (50 + i*10, 50), (150 + i*10, 150), (255, 0, 0), -1)
        cv2.circle(frame, (300 + i*5, 200), 30, (0, 255, 0), -1)
        
        synthetic_frames.append(frame)
    
    # Process synthetic frames
    print("🔄 Processing synthetic frames...")
    for i, frame in enumerate(synthetic_frames):
        annotated_frame, detections, fps = detector.process_frame(frame)
        print(f"Frame {i+1}: {len(detections)} objects detected, {fps:.1f} FPS")
    
    # Show performance plot
    print("\n📊 Generating performance visualization...")
    detector.create_performance_plot()
    
    print("\n✅ Real-time detection demo complete!")
    print("\n💡 Real-time Features:")
    print("  - Live webcam processing")
    print("  - Video file batch processing")
    print("  - FPS monitoring and optimization")
    print("  - Performance visualization")
    print("  - Screenshot capture during live detection")
    
    print("\n🎥 To use webcam detection:")
    print("  detector.webcam_detection(camera_index=0)")
    print("\n📹 To process video files:")
    print("  detector.video_file_detection('input.mp4', 'output.mp4')")

if __name__ == "__main__":
    main()
