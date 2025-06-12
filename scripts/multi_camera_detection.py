import cv2
import torch
import numpy as np
import time
import threading
from collections import deque

class MultiCameraDetection:
    def __init__(self, model_name='yolov5s', confidence_threshold=0.25):
        """
        Multi-camera object detection system
        
        Args:
            model_name: YOLOv5 model variant
            confidence_threshold: Detection confidence threshold
        """
        print("🚀 Initializing Multi-Camera Detection System...")
        
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.conf = confidence_threshold
        self.classes = self.model.names
        
        # Generate colors for classes
        np.random.seed(42)
        self.colors = {}
        for class_id, class_name in self.classes.items():
            self.colors[class_name] = tuple(map(int, np.random.randint(50, 255, 3)))
        
        # Camera management
        self.cameras = {}
        self.camera_threads = {}
        self.running = False
        
        print(f"✅ Multi-camera system initialized!")
        print(f"🎯 Model: {model_name}")
        print(f"📊 Confidence threshold: {confidence_threshold}")
    
    def detect_available_cameras(self, max_cameras=5):
        """
        Detect available cameras on the system
        
        Args:
            max_cameras: Maximum number of cameras to check
            
        Returns:
            list: Available camera indices
        """
        print("🔍 Scanning for available cameras...")
        available_cameras = []
        
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(i)
                    height, width = frame.shape[:2]
                    print(f"📷 Camera {i}: {width}x{height}")
                cap.release()
        
        if not available_cameras:
            print("❌ No cameras detected!")
        else:
            print(f"✅ Found {len(available_cameras)} camera(s): {available_cameras}")
        
        return available_cameras
    
    def setup_camera(self, camera_index, window_name):
        """
        Setup individual camera
        
        Args:
            camera_index: Camera device index
            window_name: Window name for display
            
        Returns:
            bool: Success status
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Failed to open camera {camera_index}")
            return False
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.cameras[camera_index] = {
            'capture': cap,
            'window_name': window_name,
            'fps_history': deque(maxlen=30),
            'detection_count': 0,
            'total_detections': 0
        }
        
        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        
        print(f"✅ Camera {camera_index} setup complete")
        return True
    
    def process_camera_feed(self, camera_index):
        """
        Process individual camera feed in separate thread
        
        Args:
            camera_index: Camera device index
        """
        camera_info = self.cameras[camera_index]
        cap = camera_info['capture']
        window_name = camera_info['window_name']
        
        print(f"🎬 Starting processing for camera {camera_index}")
        
        while self.running:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print(f"❌ Failed to read from camera {camera_index}")
                break
            
            # Detect objects
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            with torch.no_grad():
                results = self.model(rgb_frame)
            
            # Draw detections
            annotated_frame = self.draw_detections(frame, results, camera_index)
            
            # Calculate FPS
            end_time = time.time()
            fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            camera_info['fps_history'].append(fps)
            
            # Update detection count
            detections = results.pandas().xyxy[0]
            camera_info['detection_count'] = len(detections)
            camera_info['total_detections'] += len(detections)
            
            # Draw info panel
            self.draw_camera_info(annotated_frame, camera_index, fps, len(detections))
            
            # Display frame
            cv2.imshow(window_name, annotated_frame)
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
    
    def draw_detections(self, frame, results, camera_index):
        """Draw detections on frame"""
        annotated_frame = frame.copy()
        detections = results.pandas().xyxy[0]
        
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
    
    def draw_camera_info(self, frame, camera_index, fps, detection_count):
        """Draw camera information on frame"""
        # Info panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Camera info
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Camera {camera_index}", (20, 30), font, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50), font, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Objects: {detection_count}", (20, 70), font, 0.5, (0, 255, 0), 2)
    
    def start_multi_camera_detection(self, camera_indices=None):
        """
        Start multi-camera detection
        
        Args:
            camera_indices: List of camera indices to use (None for auto-detect)
        """
        if camera_indices is None:
            camera_indices = self.detect_available_cameras()
        
        if not camera_indices:
            print("❌ No cameras available for detection")
            return False
        
        print(f"\n🎥 Starting multi-camera detection with cameras: {camera_indices}")
        
        # Setup cameras
        for i, camera_index in enumerate(camera_indices):
            window_name = f"Camera {camera_index} - YOLOv5 Detection"
            if not self.setup_camera(camera_index, window_name):
                continue
        
        if not self.cameras:
            print("❌ No cameras could be initialized")
            return False
        
        print(f"✅ {len(self.cameras)} camera(s) initialized successfully")
        print("\n📋 Controls:")
        print("  Q - Quit all cameras")
        print("  S - Save screenshots from all cameras")
        print("=" * 60)
        
        # Start processing threads
        self.running = True
        
        for camera_index in self.cameras.keys():
            thread = threading.Thread(
                target=self.process_camera_feed,
                args=(camera_index,),
                daemon=True
            )
            thread.start()
            self.camera_threads[camera_index] = thread
        
        # Main control loop
        screenshot_count = 0
        
        try:
            while self.running:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n⏹️  Stopping all cameras...")
                    break
                elif key == ord('s') or key == ord('S'):
                    screenshot_count += 1
                    self.save_all_screenshots(screenshot_count)
                
                # Check if any window was closed
                if not any(cv2.getWindowProperty(info['window_name'], cv2.WND_PROP_VISIBLE) >= 0 
                          for info in self.cameras.values()):
                    break
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n⏹️  Detection interrupted by user")
        
        finally:
            self.cleanup()
        
        return True
    
    def save_all_screenshots(self, screenshot_count):
        """Save screenshots from all cameras"""
        print(f"📸 Saving screenshots #{screenshot_count}...")
        
        for camera_index, camera_info in self.cameras.items():
            cap = camera_info['capture']
            ret, frame = cap.read()
            
            if ret:
                # Get current detections for this frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    results = self.model(rgb_frame)
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, results, camera_index)
                
                # Save screenshot
                filename = f"camera_{camera_index}_screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"  💾 Saved: {filename}")
    
    def cleanup(self):
        """Cleanup resources"""
        print("🧹 Cleaning up resources...")
        
        self.running = False
        
        # Wait for threads to finish
        for thread in self.camera_threads.values():
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        # Release cameras
        for camera_info in self.cameras.values():
            camera_info['capture'].release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Print final statistics
        self.print_final_statistics()
        
        print("✅ Cleanup complete")
    
    def print_final_statistics(self):
        """Print final statistics for all cameras"""
        print("\n" + "=" * 60)
        print("📊 MULTI-CAMERA DETECTION STATISTICS")
        print("=" * 60)
        
        total_detections = 0
        
        for camera_index, camera_info in self.cameras.items():
            total_detections += camera_info['total_detections']
            avg_fps = np.mean(list(camera_info['fps_history'])) if camera_info['fps_history'] else 0
            
            print(f"📷 Camera {camera_index}:")
            print(f"  🎯 Total detections: {camera_info['total_detections']}")
            print(f"  ⚡ Average FPS: {avg_fps:.1f}")
        
        print(f"\n🎯 Total detections across all cameras: {total_detections}")
        print("=" * 60)

def main():
    """Main function for multi-camera detection"""
    print("🚀 YOLOv5 Multi-Camera Object Detection")
    print("=" * 60)
    
    # Initialize multi-camera detector
    detector = MultiCameraDetection(
        model_name='yolov5s',
        confidence_threshold=0.25
    )
    
    # Start multi-camera detection
    success = detector.start_multi_camera_detection()
    
    if not success:
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure cameras are connected and not used by other apps")
        print("2. Check camera permissions")
        print("3. Try running single camera detection first")
    
    print("\n👋 Multi-camera detection session ended!")

if __name__ == "__main__":
    main()
