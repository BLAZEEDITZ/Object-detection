import cv2
import torch
import numpy as np
import time
from collections import deque
import threading
import queue

class LiveCameraDetection:
    def __init__(self, model_name='yolov5s', confidence_threshold=0.25):
        """
        Real-time camera object detection with popup window
        
        Args:
            model_name: YOLOv5 model variant ('yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
            confidence_threshold: Minimum confidence for detections
        """
        print("🚀 Initializing Live Camera Detection...")
        print("📦 Loading YOLOv5 model...")
        
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.conf = confidence_threshold
        self.classes = self.model.names
        
        print(f"✅ Model {model_name} loaded successfully!")
        print(f"🎯 Confidence threshold: {confidence_threshold}")
        print(f"📋 Can detect {len(self.classes)} different object classes")
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.frame_times = deque(maxlen=30)
        
        # Detection statistics
        self.total_detections = 0
        self.detection_history = {}
        
        # Generate colors for each class
        np.random.seed(42)
        self.colors = {}
        for class_id, class_name in self.classes.items():
            self.colors[class_name] = tuple(map(int, np.random.randint(50, 255, 3)))
        
        # Camera settings
        self.camera_width = 1280
        self.camera_height = 720
        self.display_width = 1024
        self.display_height = 576
        
        # Control flags
        self.running = False
        self.paused = False
        self.show_fps = True
        self.show_confidence = True
        self.save_detections = False
        
    def detect_objects(self, frame):
        """
        Detect objects in a single frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            results: YOLOv5 detection results
        """
        # Convert BGR to RGB for YOLOv5
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection
        with torch.no_grad():
            results = self.model(rgb_frame)
        
        return results
    
    def draw_detections(self, frame, results):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            results: YOLOv5 detection results
            
        Returns:
            annotated_frame: Frame with detections drawn
        """
        annotated_frame = frame.copy()
        
        # Get detections
        detections = results.pandas().xyxy[0]
        
        # Update statistics
        self.total_detections += len(detections)
        
        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            confidence = detection['confidence']
            class_name = detection['name']
            
            # Update detection history
            if class_name in self.detection_history:
                self.detection_history[class_name] += 1
            else:
                self.detection_history[class_name] = 1
            
            # Get color for this class
            color = self.colors.get(class_name, (0, 255, 0))
            
            # Draw bounding box with thickness based on confidence
            thickness = max(2, int(confidence * 4))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            if self.show_confidence:
                label = f"{class_name}: {confidence:.2f}"
            else:
                label = class_name
            
            # Calculate label size and position
            font_scale = 0.6
            font_thickness = 2
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw label background
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + label_height + 10
            cv2.rectangle(
                annotated_frame,
                (x1, label_y - label_height - baseline),
                (x1 + label_width, label_y + baseline),
                color,
                -1
            )
            
            # Draw label text
            text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
            cv2.putText(
                annotated_frame,
                label,
                (x1, label_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                font_thickness
            )
        
        return annotated_frame, len(detections)
    
    def draw_info_panel(self, frame, fps, detection_count):
        """
        Draw information panel on frame
        
        Args:
            frame: Input frame
            fps: Current FPS
            detection_count: Number of objects detected in current frame
        """
        # Panel background
        panel_height = 120
        panel_color = (0, 0, 0, 180)  # Semi-transparent black
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, panel_height), panel_color[:3], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (0, 255, 0)
        
        # FPS
        if self.show_fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_text, (20, 35), font, font_scale, text_color, font_thickness)
        
        # Detection count
        detection_text = f"Objects: {detection_count}"
        cv2.putText(frame, detection_text, (20, 60), font, font_scale, text_color, font_thickness)
        
        # Total detections
        total_text = f"Total: {self.total_detections}"
        cv2.putText(frame, total_text, (20, 85), font, font_scale, text_color, font_thickness)
        
        # Status
        status = "PAUSED" if self.paused else "LIVE"
        status_color = (0, 255, 255) if self.paused else (0, 255, 0)
        cv2.putText(frame, status, (300, 35), font, font_scale, status_color, font_thickness)
    
    def draw_controls_help(self, frame):
        """Draw control instructions on frame"""
        help_text = [
            "Controls:",
            "Q - Quit",
            "P - Pause/Resume",
            "S - Screenshot",
            "F - Toggle FPS",
            "C - Toggle Confidence",
            "R - Reset Stats"
        ]
        
        # Help panel background
        panel_width = 200
        panel_height = len(help_text) * 25 + 20
        panel_x = frame.shape[1] - panel_width - 10
        panel_y = 10
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw help text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (255, 255, 255)
        
        for i, text in enumerate(help_text):
            y_pos = panel_y + 20 + i * 25
            cv2.putText(frame, text, (panel_x + 10, y_pos), 
                       font, font_scale, text_color, font_thickness)
    
    def start_camera_detection(self, camera_index=0, show_help=True):
        """
        Start real-time camera detection with popup window
        
        Args:
            camera_index: Camera device index (0 for default camera)
            show_help: Whether to show control instructions
        """
        print(f"\n🎥 Starting camera detection...")
        print(f"📹 Using camera index: {camera_index}")
        print("=" * 60)
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_index}")
            print("💡 Try different camera indices (0, 1, 2, etc.)")
            return False
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual camera properties
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📷 Camera resolution: {actual_width}x{actual_height}")
        print(f"🎬 Camera FPS: {actual_fps}")
        print(f"🖥️  Display resolution: {self.display_width}x{self.display_height}")
        
        # Create window
        window_name = "YOLOv5 Live Object Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.display_width, self.display_height)
        
        print("\n🎯 Detection started! Camera window should appear...")
        print("📋 Controls:")
        print("  Q - Quit detection")
        print("  P - Pause/Resume")
        print("  S - Save screenshot")
        print("  F - Toggle FPS display")
        print("  C - Toggle confidence display")
        print("  R - Reset statistics")
        print("=" * 60)
        
        self.running = True
        screenshot_count = 0
        frame_count = 0
        
        try:
            while self.running:
                start_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error reading from camera")
                    break
                
                frame_count += 1
                
                if not self.paused:
                    # Detect objects
                    results = self.detect_objects(frame)
                    
                    # Draw detections
                    annotated_frame, detection_count = self.draw_detections(frame, results)
                else:
                    annotated_frame = frame
                    detection_count = 0
                
                # Calculate FPS
                end_time = time.time()
                frame_time = end_time - start_time
                fps = 1.0 / frame_time if frame_time > 0 else 0
                
                self.fps_history.append(fps)
                self.frame_times.append(frame_time)
                
                # Draw information panel
                self.draw_info_panel(annotated_frame, fps, detection_count)
                
                # Draw help panel
                if show_help:
                    self.draw_controls_help(annotated_frame)
                
                # Resize frame for display
                if annotated_frame.shape[1] != self.display_width or annotated_frame.shape[0] != self.display_height:
                    annotated_frame = cv2.resize(annotated_frame, (self.display_width, self.display_height))
                
                # Display frame
                cv2.imshow(window_name, annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n⏹️  Stopping detection...")
                    break
                elif key == ord('p') or key == ord('P'):
                    self.paused = not self.paused
                    status = "PAUSED" if self.paused else "RESUMED"
                    print(f"⏸️  Detection {status}")
                elif key == ord('s') or key == ord('S'):
                    screenshot_count += 1
                    filename = f"detection_screenshot_{screenshot_count:03d}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('f') or key == ord('F'):
                    self.show_fps = not self.show_fps
                    print(f"📊 FPS display: {'ON' if self.show_fps else 'OFF'}")
                elif key == ord('c') or key == ord('C'):
                    self.show_confidence = not self.show_confidence
                    print(f"🎯 Confidence display: {'ON' if self.show_confidence else 'OFF'}")
                elif key == ord('r') or key == ord('R'):
                    self.reset_statistics()
                    print("🔄 Statistics reset")
                
                # Print periodic updates
                if frame_count % 100 == 0:
                    avg_fps = np.mean(list(self.fps_history)) if self.fps_history else 0
                    print(f"📈 Frame {frame_count}: Avg FPS = {avg_fps:.1f}")
        
        except KeyboardInterrupt:
            print("\n⏹️  Detection interrupted by user")
        
        except Exception as e:
            print(f"❌ Error during detection: {e}")
        
        finally:
            # Cleanup
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self.print_final_statistics(frame_count)
        
        return True
    
    def reset_statistics(self):
        """Reset detection statistics"""
        self.total_detections = 0
        self.detection_history = {}
        self.fps_history.clear()
        self.frame_times.clear()
    
    def print_final_statistics(self, total_frames):
        """Print final detection statistics"""
        print("\n" + "=" * 60)
        print("📊 FINAL DETECTION STATISTICS")
        print("=" * 60)
        
        print(f"🎬 Total frames processed: {total_frames}")
        print(f"🎯 Total objects detected: {self.total_detections}")
        
        if self.fps_history:
            avg_fps = np.mean(list(self.fps_history))
            min_fps = np.min(list(self.fps_history))
            max_fps = np.max(list(self.fps_history))
            print(f"⚡ Average FPS: {avg_fps:.1f}")
            print(f"📉 Min FPS: {min_fps:.1f}")
            print(f"📈 Max FPS: {max_fps:.1f}")
        
        if self.detection_history:
            print(f"\n🏷️  Objects detected by class:")
            sorted_detections = sorted(self.detection_history.items(), 
                                     key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_detections[:10]:  # Top 10
                percentage = (count / self.total_detections) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        print("=" * 60)
        print("✅ Detection session completed!")

def main():
    """Main function to start live camera detection"""
    print("🚀 YOLOv5 Live Camera Object Detection")
    print("=" * 60)
    
    # Initialize detector
    print("🔧 Initializing detector...")
    detector = LiveCameraDetection(
        model_name='yolov5s',  # Use 'yolov5s' for speed, 'yolov5m' or 'yolov5l' for better accuracy
        confidence_threshold=0.25
    )
    
    # Start camera detection
    success = detector.start_camera_detection(
        camera_index=0,  # Try 0, 1, 2, etc. if camera doesn't work
        show_help=True
    )
    
    if not success:
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure your camera is connected and not used by other apps")
        print("2. Try different camera indices (0, 1, 2, etc.)")
        print("3. Check camera permissions")
        print("4. Restart the script")
    
    print("\n👋 Thank you for using YOLOv5 Live Detection!")

if __name__ == "__main__":
    main()
