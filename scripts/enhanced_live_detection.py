import cv2
import torch
import numpy as np
import time
from collections import deque, defaultdict
import json
import os
from datetime import datetime

class EnhancedLiveDetection:
    def __init__(self, model_name='yolov5s', confidence_threshold=0.25):
        """
        Enhanced live detection with advanced features
        """
        print("🚀 Initializing Enhanced Live Detection...")
        
        # Load model
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.conf = confidence_threshold
        self.classes = self.model.names
        
        # Enhanced features
        self.object_tracker = ObjectTracker()
        self.alert_system = AlertSystem()
        self.recording_system = RecordingSystem()
        
        # Performance monitoring
        self.fps_history = deque(maxlen=60)
        self.detection_history = defaultdict(int)
        self.session_stats = {
            'start_time': None,
            'total_frames': 0,
            'total_detections': 0,
            'unique_objects': set()
        }
        
        # Visual settings
        np.random.seed(42)
        self.colors = {}
        for class_id, class_name in self.classes.items():
            self.colors[class_name] = tuple(map(int, np.random.randint(50, 255, 3)))
        
        # Control settings
        self.show_trails = False
        self.show_zones = False
        self.recording = False
        self.alerts_enabled = True
        
        print("✅ Enhanced detection system ready!")
    
    def start_enhanced_detection(self, camera_index=0):
        """Start enhanced live detection with all features"""
        print(f"\n🎥 Starting Enhanced Live Detection...")
        print("=" * 60)
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_index}")
            return False
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📷 Camera: {width}x{height} @ {fps}fps")
        
        # Setup window
        window_name = "Enhanced YOLOv5 Live Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 576)
        
        # Initialize systems
        self.recording_system.setup(width, height, fps)
        self.session_stats['start_time'] = datetime.now()
        
        print("\n🎯 Enhanced Detection Active!")
        print("📋 Enhanced Controls:")
        print("  Q - Quit")
        print("  SPACE - Pause/Resume")
        print("  R - Start/Stop Recording")
        print("  T - Toggle Object Trails")
        print("  Z - Toggle Detection Zones")
        print("  A - Toggle Alerts")
        print("  S - Save Screenshot")
        print("  C - Clear Statistics")
        print("=" * 60)
        
        paused = False
        screenshot_count = 0
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.session_stats['total_frames'] += 1
                
                if not paused:
                    # Detect objects
                    results = self.detect_objects(frame)
                    detections = results.pandas().xyxy[0]
                    
                    # Update tracker
                    tracked_objects = self.object_tracker.update(detections)
                    
                    # Check alerts
                    if self.alerts_enabled:
                        self.alert_system.check_alerts(tracked_objects, frame.shape)
                    
                    # Draw everything
                    annotated_frame = self.draw_enhanced_detections(
                        frame, detections, tracked_objects
                    )
                    
                    # Update statistics
                    self.update_statistics(detections)
                    
                else:
                    annotated_frame = frame.copy()
                    cv2.putText(annotated_frame, "PAUSED", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                
                # Calculate FPS
                end_time = time.time()
                fps_current = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
                self.fps_history.append(fps_current)
                
                # Draw UI
                self.draw_enhanced_ui(annotated_frame, fps_current)
                
                # Record if enabled
                if self.recording:
                    self.recording_system.write_frame(annotated_frame)
                
                # Display
                cv2.imshow(window_name, annotated_frame)
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord(' '):  # Space bar
                    paused = not paused
                    print(f"⏸️  {'PAUSED' if paused else 'RESUMED'}")
                elif key == ord('r') or key == ord('R'):
                    if self.recording:
                        self.recording_system.stop_recording()
                        self.recording = False
                        print("⏹️  Recording stopped")
                    else:
                        self.recording_system.start_recording()
                        self.recording = True
                        print("🔴 Recording started")
                elif key == ord('t') or key == ord('T'):
                    self.show_trails = not self.show_trails
                    print(f"👣 Object trails: {'ON' if self.show_trails else 'OFF'}")
                elif key == ord('z') or key == ord('Z'):
                    self.show_zones = not self.show_zones
                    print(f"🎯 Detection zones: {'ON' if self.show_zones else 'OFF'}")
                elif key == ord('a') or key == ord('A'):
                    self.alerts_enabled = not self.alerts_enabled
                    print(f"🚨 Alerts: {'ON' if self.alerts_enabled else 'OFF'}")
                elif key == ord('s') or key == ord('S'):
                    screenshot_count += 1
                    filename = f"enhanced_screenshot_{screenshot_count:03d}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('c') or key == ord('C'):
                    self.clear_statistics()
                    print("🔄 Statistics cleared")
        
        except KeyboardInterrupt:
            print("\n⏹️  Detection stopped by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            if self.recording:
                self.recording_system.stop_recording()
            
            self.print_session_summary()
        
        return True
    
    def detect_objects(self, frame):
        """Detect objects with timing"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            results = self.model(rgb_frame)
        return results
    
    def draw_enhanced_detections(self, frame, detections, tracked_objects):
        """Draw enhanced detections with tracking and trails"""
        annotated_frame = frame.copy()
        
        # Draw detection zones if enabled
        if self.show_zones:
            self.draw_detection_zones(annotated_frame)
        
        # Draw object trails if enabled
        if self.show_trails:
            self.object_tracker.draw_trails(annotated_frame)
        
        # Draw current detections
        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            confidence = detection['confidence']
            class_name = detection['name']
            
            # Get color
            color = self.colors.get(class_name, (0, 255, 0))
            
            # Draw bounding box with confidence-based thickness
            thickness = max(2, int(confidence * 4))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw center point
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(annotated_frame, (center_x, center_y), 3, color, -1)
            
            # Enhanced label with ID if tracked
            object_id = self.get_object_id(detection, tracked_objects)
            if object_id is not None:
                label = f"ID:{object_id} {class_name}: {confidence:.2f}"
            else:
                label = f"{class_name}: {confidence:.2f}"
            
            # Draw label background
            font_scale = 0.6
            font_thickness = 2
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10),
                         (x1 + label_width, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        return annotated_frame
    
    def draw_enhanced_ui(self, frame, fps):
        """Draw enhanced UI with statistics and controls"""
        h, w = frame.shape[:2]
        
        # Main info panel
        panel_height = 150
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Statistics
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)
        
        y_pos = 30
        cv2.putText(frame, f"FPS: {fps:.1f} (Avg: {np.mean(list(self.fps_history)):.1f})", 
                   (20, y_pos), font, font_scale, color, 2)
        
        y_pos += 20
        cv2.putText(frame, f"Frame: {self.session_stats['total_frames']}", 
                   (20, y_pos), font, font_scale, color, 2)
        
        y_pos += 20
        cv2.putText(frame, f"Total Detections: {self.session_stats['total_detections']}", 
                   (20, y_pos), font, font_scale, color, 2)
        
        y_pos += 20
        cv2.putText(frame, f"Unique Objects: {len(self.session_stats['unique_objects'])}", 
                   (20, y_pos), font, font_scale, color, 2)
        
        # Status indicators
        y_pos += 25
        status_color = (0, 255, 0) if self.recording else (100, 100, 100)
        cv2.putText(frame, f"REC: {'ON' if self.recording else 'OFF'}", 
                   (20, y_pos), font, font_scale, status_color, 2)
        
        cv2.putText(frame, f"TRAILS: {'ON' if self.show_trails else 'OFF'}", 
                   (100, y_pos), font, font_scale, color, 2)
        
        y_pos += 20
        cv2.putText(frame, f"ALERTS: {'ON' if self.alerts_enabled else 'OFF'}", 
                   (20, y_pos), font, font_scale, color, 2)
        
        # Top detections panel
        if self.detection_history:
            top_detections = sorted(self.detection_history.items(), 
                                  key=lambda x: x[1], reverse=True)[:5]
            
            panel_x = w - 250
            panel_y = 10
            panel_width = 240
            panel_height = 30 + len(top_detections) * 20
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), 
                         (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            cv2.putText(frame, "Top Detections:", (panel_x + 10, panel_y + 20), 
                       font, font_scale, (255, 255, 255), 2)
            
            for i, (class_name, count) in enumerate(top_detections):
                y = panel_y + 40 + i * 20
                cv2.putText(frame, f"{class_name}: {count}", (panel_x + 10, y), 
                           font, font_scale, self.colors.get(class_name, (255, 255, 255)), 2)
    
    def draw_detection_zones(self, frame):
        """Draw detection zones on frame"""
        h, w = frame.shape[:2]
        
        # Draw center zone
        center_x, center_y = w // 2, h // 2
        zone_size = min(w, h) // 4
        
        cv2.rectangle(frame, 
                     (center_x - zone_size, center_y - zone_size),
                     (center_x + zone_size, center_y + zone_size),
                     (255, 255, 0), 2)
        cv2.putText(frame, "CENTER ZONE", (center_x - 60, center_y - zone_size - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    def get_object_id(self, detection, tracked_objects):
        """Get object ID from tracker"""
        # Simplified - in real implementation, match detection to tracked object
        return None
    
    def update_statistics(self, detections):
        """Update session statistics"""
        self.session_stats['total_detections'] += len(detections)
        
        for _, detection in detections.iterrows():
            class_name = detection['name']
            self.detection_history[class_name] += 1
            self.session_stats['unique_objects'].add(class_name)
    
    def clear_statistics(self):
        """Clear all statistics"""
        self.session_stats['total_detections'] = 0
        self.session_stats['unique_objects'].clear()
        self.detection_history.clear()
        self.fps_history.clear()
    
    def print_session_summary(self):
        """Print final session summary"""
        duration = datetime.now() - self.session_stats['start_time']
        
        print("\n" + "=" * 60)
        print("📊 ENHANCED DETECTION SESSION SUMMARY")
        print("=" * 60)
        print(f"⏱️  Duration: {duration}")
        print(f"🎬 Total Frames: {self.session_stats['total_frames']}")
        print(f"🎯 Total Detections: {self.session_stats['total_detections']}")
        print(f"🏷️  Unique Object Types: {len(self.session_stats['unique_objects'])}")
        
        if self.fps_history:
            avg_fps = np.mean(list(self.fps_history))
            print(f"⚡ Average FPS: {avg_fps:.1f}")
        
        if self.detection_history:
            print("\n🏆 Top Detected Objects:")
            top_objects = sorted(self.detection_history.items(), 
                               key=lambda x: x[1], reverse=True)[:10]
            for class_name, count in top_objects:
                print(f"  {class_name}: {count}")
        
        print("=" * 60)

# Helper classes
class ObjectTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.trail_length = 30
    
    def update(self, detections):
        # Simplified tracker - in real implementation, use proper tracking algorithm
        return {}
    
    def draw_trails(self, frame):
        # Draw object movement trails
        pass

class AlertSystem:
    def __init__(self):
        self.alerts = []
    
    def check_alerts(self, tracked_objects, frame_shape):
        # Check for various alert conditions
        pass

class RecordingSystem:
    def __init__(self):
        self.writer = None
        self.recording = False
    
    def setup(self, width, height, fps):
        self.width = width
        self.height = height
        self.fps = fps
    
    def start_recording(self):
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_recording_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(filename, fourcc, self.fps, (self.width, self.height))
            self.recording = True
            print(f"🎥 Recording started: {filename}")
    
    def write_frame(self, frame):
        if self.recording and self.writer:
            self.writer.write(frame)
    
    def stop_recording(self):
        if self.recording and self.writer:
            self.writer.release()
            self.writer = None
            self.recording = False
            print("⏹️  Recording stopped")

def main():
    """Main function for enhanced live detection"""
    print("🚀 Enhanced YOLOv5 Live Detection System")
    print("=" * 60)
    
    # Initialize enhanced detector
    detector = EnhancedLiveDetection(
        model_name='yolov5s',
        confidence_threshold=0.25
    )
    
    # Start enhanced detection
    success = detector.start_enhanced_detection(camera_index=0)
    
    if not success:
        print("\n💡 Troubleshooting:")
        print("1. Check camera connection")
        print("2. Try different camera index")
        print("3. Check permissions")
    
    print("\n👋 Enhanced detection session ended!")

if __name__ == "__main__":
    main()
