import torch
import cv2
import numpy as np
import time
from pathlib import Path
import json
import threading
import queue
from collections import deque

class OptimizedRealTimeDetector:
    def __init__(self, model_path, optimization_config=None):
        """
        Optimized real-time detector using custom trained model
        
        Args:
            model_path: Path to optimized custom model
            optimization_config: Configuration from optimization results
        """
        self.model_path = Path(model_path)
        self.optimization_config = optimization_config or {}
        
        # Load optimized model
        self.model = self.load_optimized_model()
        
        # Apply optimization settings
        self.apply_optimization_settings()
        
        # Performance tracking
        self.fps_history = deque(maxlen=60)
        self.detection_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'class_counts': {},
            'confidence_distribution': []
        }
        
        # Real-time processing
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        self.processing_thread = None
        self.running = False
        
        print(f"🚀 Optimized Real-Time Detector initialized")
        print(f"📁 Model: {self.model_path}")
    
    def load_optimized_model(self):
        """Load the optimized custom model"""
        try:
            # Try loading TorchScript model first
            torchscript_path = self.model_path.parent / "optimized_model.torchscript"
            if torchscript_path.exists():
                print("📦 Loading TorchScript optimized model...")
                model = torch.jit.load(str(torchscript_path))
                model.eval()
                return model
            
            # Fallback to regular PyTorch model
            print("📦 Loading PyTorch model...")
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(self.model_path))
            return model
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None
    
    def apply_optimization_settings(self):
        """Apply optimization settings from configuration"""
        if not self.optimization_config:
            # Load optimization config if available
            config_path = self.model_path.parent / "optimization_report.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.optimization_config = json.load(f)
        
        # Apply optimal confidence threshold
        if 'confidence_threshold' in self.optimization_config:
            optimal_conf = self.optimization_config['confidence_threshold'].get('optimal', 0.25)
            if hasattr(self.model, 'conf'):
                self.model.conf = optimal_conf
                print(f"🎯 Applied optimal confidence threshold: {optimal_conf}")
        
        # Apply optimal NMS threshold
        if 'nms_threshold' in self.optimization_config:
            # Find optimal NMS threshold (highest FPS with reasonable detections)
            nms_results = self.optimization_config['nms_threshold']
            optimal_nms = max(nms_results.keys(), key=lambda k: nms_results[k]['fps'])
            if hasattr(self.model, 'iou'):
                self.model.iou = float(optimal_nms)
                print(f"🔧 Applied optimal NMS threshold: {optimal_nms}")
    
    def process_frame_threaded(self, frame):
        """Process frame in separate thread for better performance"""
        if not self.frame_queue.full():
            self.frame_queue.put(frame)
    
    def detection_worker(self):
        """Worker thread for processing frames"""
        while self.running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=0.1)
                
                # Process frame
                start_time = time.time()
                results = self.model(frame)
                end_time = time.time()
                
                # Calculate FPS
                fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
                
                # Put result in queue
                if not self.result_queue.full():
                    self.result_queue.put({
                        'frame': frame,
                        'results': results,
                        'fps': fps,
                        'timestamp': time.time()
                    })
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Detection worker error: {e}")
    
    def start_optimized_detection(self, camera_index=0, use_threading=True):
        """Start optimized real-time detection"""
        print(f"🎥 Starting optimized real-time detection...")
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_index}")
            return False
        
        # Set camera properties for optimal performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        # Create window
        window_name = "Optimized YOLOv5 Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 576)
        
        # Start processing thread if enabled
        if use_threading:
            self.running = True
            self.processing_thread = threading.Thread(target=self.detection_worker, daemon=True)
            self.processing_thread.start()
            print("🔄 Multi-threaded processing enabled")
        
        print("\n📋 Controls:")
        print("  Q - Quit")
        print("  S - Save screenshot")
        print("  R - Reset statistics")
        print("  T - Toggle threading")
        print("=" * 60)
        
        screenshot_count = 0
        last_result = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.detection_stats['total_frames'] += 1
                
                if use_threading:
                    # Threaded processing
                    self.process_frame_threaded(frame)
                    
                    # Get latest result
                    try:
                        result_data = self.result_queue.get_nowait()
                        last_result = result_data
                        self.fps_history.append(result_data['fps'])
                    except queue.Empty:
                        pass
                    
                    # Use last result for display
                    if last_result:
                        annotated_frame = self.draw_optimized_detections(
                            frame, last_result['results'], last_result['fps']
                        )
                    else:
                        annotated_frame = frame
                
                else:
                    # Direct processing
                    start_time = time.time()
                    results = self.model(frame)
                    end_time = time.time()
                    
                    fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
                    self.fps_history.append(fps)
                    
                    annotated_frame = self.draw_optimized_detections(frame, results, fps)
                
                # Display frame
                cv2.imshow(window_name, annotated_frame)
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    screenshot_count += 1
                    filename = f"optimized_detection_{screenshot_count:03d}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('r') or key == ord('R'):
                    self.reset_statistics()
                    print("🔄 Statistics reset")
                elif key == ord('t') or key == ord('T'):
                    use_threading = not use_threading
                    if use_threading and not self.running:
                        self.running = True
                        self.processing_thread = threading.Thread(target=self.detection_worker, daemon=True)
                        self.processing_thread.start()
                    elif not use_threading:
                        self.running = False
                    print(f"🔄 Threading: {'ON' if use_threading else 'OFF'}")
        
        except KeyboardInterrupt:
            print("\n⏹️  Detection stopped by user")
        
        finally:
            # Cleanup
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
            
            self.print_performance_summary()
        
        return True
    
    def draw_optimized_detections(self, frame, results, fps):
        """Draw detections with optimized visualization"""
        annotated_frame = frame.copy()
        
        # Get detections
        detections = results.pandas().xyxy[0] if hasattr(results, 'pandas') else []
        
        if len(detections) > 0:
            self.detection_stats['total_detections'] += len(detections)
            
            # Draw detections
            for _, detection in detections.iterrows():
                x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                confidence = detection['confidence']
                class_name = detection['name']
                
                # Update statistics
                self.detection_stats['class_counts'][class_name] = self.detection_stats['class_counts'].get(class_name, 0) + 1
                self.detection_stats['confidence_distribution'].append(confidence)
                
                # Dynamic color based on confidence
                confidence_color = int(255 * confidence)
                color = (0, confidence_color, 255 - confidence_color)
                
                # Draw bounding box
                thickness = max(1, int(confidence * 3))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Optimized label drawing
                label = f"{class_name}: {confidence:.2f}"
                font_scale = 0.5
                font_thickness = 1
                
                # Get label size
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                
                # Draw label background
                cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10),
                             (x1 + label_width, y1), color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        # Draw optimized UI
        self.draw_optimized_ui(annotated_frame, fps, len(detections))
        
        return annotated_frame
    
    def draw_optimized_ui(self, frame, fps, detection_count):
        """Draw optimized UI overlay"""
        h, w = frame.shape[:2]
        
        # Performance panel
        panel_height = 100
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Performance metrics
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)
        
        # Current FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), font, font_scale, color, 2)
        
        # Average FPS
        if self.fps_history:
            avg_fps = np.mean(list(self.fps_history))
            cv2.putText(frame, f"Avg FPS: {avg_fps:.1f}", (20, 50), font, font_scale, color, 2)
        
        # Detection count
        cv2.putText(frame, f"Objects: {detection_count}", (20, 70), font, font_scale, color, 2)
        
        # Total statistics
        cv2.putText(frame, f"Total: {self.detection_stats['total_detections']}", (20, 90), font, font_scale, color, 2)
        
        # Model info
        model_info = f"Custom YOLOv5 | Optimized"
        cv2.putText(frame, model_info, (w - 250, 30), font, font_scale, (255, 255, 255), 2)
        
        # Performance indicator
        if fps > 25:
            perf_color = (0, 255, 0)  # Green
            perf_text = "EXCELLENT"
        elif fps > 15:
            perf_color = (0, 255, 255)  # Yellow
            perf_text = "GOOD"
        else:
            perf_color = (0, 0, 255)  # Red
            perf_text = "NEEDS OPTIMIZATION"
        
        cv2.putText(frame, perf_text, (w - 200, 60), font, font_scale, perf_color, 2)
    
    def reset_statistics(self):
        """Reset detection statistics"""
        self.detection_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'class_counts': {},
            'confidence_distribution': []
        }
        self.fps_history.clear()
    
    def print_performance_summary(self):
        """Print final performance summary"""
        print("\n" + "=" * 60)
        print("📊 OPTIMIZED DETECTION PERFORMANCE SUMMARY")
        print("=" * 60)
        
        if self.fps_history:
            avg_fps = np.mean(list(self.fps_history))
            min_fps = np.min(list(self.fps_history))
            max_fps = np.max(list(self.fps_history))
            
            print(f"⚡ Performance Metrics:")
            print(f"  Average FPS: {avg_fps:.1f}")
            print(f"  Min FPS: {min_fps:.1f}")
            print(f"  Max FPS: {max_fps:.1f}")
            print(f"  FPS Stability: {np.std(list(self.fps_history)):.2f}")
        
        print(f"\n🎯 Detection Statistics:")
        print(f"  Total Frames: {self.detection_stats['total_frames']}")
        print(f"  Total Detections: {self.detection_stats['total_detections']}")
        
        if self.detection_stats['total_frames'] > 0:
            detection_rate = self.detection_stats['total_detections'] / self.detection_stats['total_frames']
            print(f"  Avg Detections/Frame: {detection_rate:.2f}")
        
        if self.detection_stats['class_counts']:
            print(f"\n🏷️  Top Detected Classes:")
            sorted_classes = sorted(self.detection_stats['class_counts'].items(), 
                                  key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_classes[:5]:
                print(f"    {class_name}: {count}")
        
        if self.detection_stats['confidence_distribution']:
            avg_confidence = np.mean(self.detection_stats['confidence_distribution'])
            print(f"\n📊 Average Confidence: {avg_confidence:.3f}")
        
        print("=" * 60)

def main():
    """Demonstrate integrated optimized detection pipeline"""
    print("🚀 Integrated YOLOv5 Custom Training & Optimization Pipeline")
    print("=" * 60)
    
    # Find trained model
    model_pattern = "runs/train/custom_yolov5s_*/weights/best.pt"
    model_files = list(Path(".").glob(model_pattern))
    
    if not model_files:
        print(f"❌ No trained model found matching: {model_pattern}")
        print("\n💡 Complete Pipeline Steps:")
        print("1. Run dataset_preparation.py to create dataset")
        print("2. Run annotation_tool.py to annotate images")
        print("3. Run training_pipeline.py to train model")
        print("4. Run model_optimization.py to optimize model")
        print("5. Run this script for optimized real-time detection")
        return
    
    # Use most recent model
    model_file = sorted(model_files)[-1]
    print(f"📁 Using trained model: {model_file}")
    
    # Initialize optimized detector
    detector = OptimizedRealTimeDetector(model_file)
    
    if detector.model is None:
        print("❌ Failed to initialize detector")
        return
    
    print("\n🎯 Custom Model Information:")
    print(f"  Model Path: {detector.model_path}")
    print(f"  Optimization Config: {'Loaded' if detector.optimization_config else 'Default'}")
    
    # Start optimized detection
    print(f"\n🎥 Starting optimized real-time detection...")
    success = detector.start_optimized_detection(camera_index=0, use_threading=True)
    
    if success:
        print("✅ Optimized detection completed successfully!")
    else:
        print("❌ Detection failed!")
    
    print("\n🎉 Custom YOLOv5 Pipeline Complete!")
    print("💡 Your custom-trained, optimized model is now ready for deployment!")

if __name__ == "__main__":
    main()
