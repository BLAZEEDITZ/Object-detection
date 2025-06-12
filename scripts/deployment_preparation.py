"""
YOLOv5 Deployment Preparation
============================

This script prepares the trained model for real-time deployment including:
- Model optimization and conversion
- Performance benchmarking
- Deployment configuration
- Integration with real-time detection system
"""

import torch
import cv2
import numpy as np
import time
import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import subprocess
import shutil
from datetime import datetime
import threading
import queue
import psutil
import platform
import sys
import os

class YOLOv5DeploymentPreparator:
    def __init__(self, project_dir):
        """
        Initialize deployment preparation
        
        Args:
            project_dir: Path to project directory
        """
        self.project_dir = Path(project_dir)
        self.models_dir = self.project_dir / "models"
        self.deployment_dir = self.project_dir / "deployment"
        self.configs_dir = self.project_dir / "configs"
        
        # Create deployment directory structure
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        (self.deployment_dir / "packages").mkdir(exist_ok=True)
        (self.deployment_dir / "configs").mkdir(exist_ok=True)
        (self.deployment_dir / "docs").mkdir(exist_ok=True)
        (self.deployment_dir / "models").mkdir(exist_ok=True)
        
        # Load configurations
        self.load_configurations()
        
        # Load trained model
        self.model = self.load_trained_model()
        
        print(f"🚀 YOLOv5 Deployment Preparator initialized")
        print(f"📁 Project: {self.project_dir}")
    
    def load_configurations(self):
        """Load project configurations"""
        # Load dataset config
        dataset_yaml = self.project_dir / "dataset" / "dataset.yaml"
        if dataset_yaml.exists():
            with open(dataset_yaml, 'r') as f:
                self.dataset_config = yaml.safe_load(f)
            self.class_names = self.dataset_config['names']
        else:
            # Create default config if not found
            self.class_names = ['person', 'car', 'bicycle']  # Default classes
            self.dataset_config = {
                'names': self.class_names,
                'nc': len(self.class_names)
            }
            print("⚠️ Dataset config not found, using defaults")
    
    def load_trained_model(self):
        """Load the trained model"""
        model_path = self.models_dir / "trained" / "best.pt"
        
        if not model_path.exists():
            # Try alternative paths
            alt_paths = [
                self.models_dir / "best.pt",
                self.project_dir / "best.pt",
                Path("best.pt")
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    model_path = alt_path
                    break
            else:
                print("⚠️ No trained model found, using YOLOv5s pretrained")
                return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        try:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path))
            print(f"✅ Model loaded: {model_path}")
            return model
        except Exception as e:
            print(f"⚠️ Failed to load custom model: {e}")
            print("Using YOLOv5s pretrained model instead")
            return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    def optimize_model(self):
        """Optimize model for deployment"""
        print("\n🔧 OPTIMIZING MODEL FOR DEPLOYMENT")
        print("-" * 50)
        
        optimized_dir = self.deployment_dir / "models" / "optimized"
        optimized_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. TorchScript optimization
        print("1. Converting to TorchScript...")
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, 640, 640)
            
            # Convert to TorchScript
            traced_model = torch.jit.trace(self.model.model, dummy_input)
            torchscript_path = optimized_dir / "model.torchscript"
            traced_model.save(str(torchscript_path))
            print(f"   ✅ TorchScript saved: {torchscript_path}")
        except Exception as e:
            print(f"   ❌ TorchScript conversion failed: {e}")
        
        # 2. Model quantization
        print("2. Applying quantization...")
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                self.model.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            quantized_path = optimized_dir / "model_quantized.pt"
            torch.save(quantized_model.state_dict(), quantized_path)
            print(f"   ✅ Quantized model saved: {quantized_path}")
        except Exception as e:
            print(f"   ❌ Quantization failed: {e}")
        
        # 3. Half precision
        print("3. Converting to half precision...")
        try:
            if torch.cuda.is_available():
                half_model = self.model.half()
                half_path = optimized_dir / "model_half.pt"
                torch.save(half_model.state_dict(), half_path)
                print(f"   ✅ Half precision model saved: {half_path}")
            else:
                print("   ⚠️ CUDA not available, skipping half precision")
        except Exception as e:
            print(f"   ❌ Half precision conversion failed: {e}")
        
        print("✅ Model optimization completed")
    
    def benchmark_performance(self):
        """Benchmark model performance"""
        print("\n📊 BENCHMARKING PERFORMANCE")
        print("-" * 50)
        
        results = {}
        
        # Test different input sizes
        input_sizes = [416, 640, 832]
        batch_sizes = [1, 4, 8] if torch.cuda.is_available() else [1]
        
        for input_size in input_sizes:
            for batch_size in batch_sizes:
                print(f"Testing {input_size}x{input_size}, batch={batch_size}...")
                
                try:
                    # Create test input
                    test_input = torch.randn(batch_size, 3, input_size, input_size)
                    
                    if torch.cuda.is_available():
                        test_input = test_input.cuda()
                        self.model.cuda()
                    
                    # Warmup
                    for _ in range(5):
                        with torch.no_grad():
                            _ = self.model(test_input)
                    
                    # Benchmark
                    times = []
                    for _ in range(20):
                        start_time = time.time()
                        with torch.no_grad():
                            _ = self.model(test_input)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        times.append(time.time() - start_time)
                    
                    avg_time = np.mean(times)
                    fps = batch_size / avg_time
                    
                    key = f"{input_size}x{input_size}_batch{batch_size}"
                    results[key] = {
                        'avg_time': avg_time,
                        'fps': fps,
                        'input_size': input_size,
                        'batch_size': batch_size
                    }
                    
                    print(f"   ⏱️ Avg time: {avg_time:.3f}s, FPS: {fps:.1f}")
                    
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
        
        # Save benchmark results
        benchmark_path = self.deployment_dir / "benchmark_results.json"
        with open(benchmark_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Benchmark results saved: {benchmark_path}")
        return results
    
    def export_model_formats(self):
        """Export model to different formats"""
        print("\n📦 EXPORTING MODEL FORMATS")
        print("-" * 50)
        
        export_dir = self.deployment_dir / "models" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. ONNX Export
        print("1. Exporting to ONNX...")
        try:
            dummy_input = torch.randn(1, 3, 640, 640)
            onnx_path = export_dir / "model.onnx"
            
            torch.onnx.export(
                self.model.model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            print(f"   ✅ ONNX model saved: {onnx_path}")
        except Exception as e:
            print(f"   ❌ ONNX export failed: {e}")
        
        # 2. TensorRT Export (if available)
        print("2. Checking TensorRT availability...")
        try:
            import tensorrt as trt
            print("   ✅ TensorRT available")
            # TensorRT export would go here
            print("   ℹ️ TensorRT export requires additional setup")
        except ImportError:
            print("   ⚠️ TensorRT not available")
        
        # 3. CoreML Export (if on macOS)
        print("3. Checking CoreML availability...")
        if platform.system() == "Darwin":  # macOS
            try:
                import coremltools as ct
                print("   ✅ CoreML available")
                # CoreML export would go here
                print("   ℹ️ CoreML export requires additional configuration")
            except ImportError:
                print("   ⚠️ CoreML not available")
        else:
            print("   ⚠️ CoreML only available on macOS")
        
        # 4. TensorFlow Lite Export
        print("4. Checking TensorFlow Lite availability...")
        try:
            import tensorflow as tf
            print("   ✅ TensorFlow available")
            # TFLite export would go here
            print("   ℹ️ TFLite export requires ONNX to TF conversion")
        except ImportError:
            print("   ⚠️ TensorFlow not available")
        
        print("✅ Model format exports completed")
    
    def create_deployment_config(self):
        """Create deployment configuration files"""
        print("\n⚙️ CREATING DEPLOYMENT CONFIGURATION")
        print("-" * 50)
        
        config_dir = self.deployment_dir / "configs"
        
        # Main deployment config
        deployment_config = {
            'model': {
                'path': 'models/best.pt',
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45,
                'input_size': [640, 640],
                'classes': self.class_names,
                'device': 'auto'  # auto, cpu, cuda:0, etc.
            },
            'deployment': {
                'type': 'standalone',  # standalone, docker, cloud, edge
                'batch_size': 1,
                'max_workers': 4,
                'enable_gpu': True,
                'optimize_for_speed': False
            },
            'input': {
                'sources': ['camera', 'video', 'image', 'stream'],
                'camera_index': 0,
                'video_formats': ['.mp4', '.avi', '.mov', '.mkv'],
                'image_formats': ['.jpg', '.jpeg', '.png', '.bmp']
            },
            'output': {
                'save_results': True,
                'output_dir': 'results',
                'save_images': True,
                'save_videos': True,
                'save_json': True
            },
            'visualization': {
                'show_labels': True,
                'show_confidence': True,
                'line_thickness': 2,
                'font_scale': 0.5
            }
        }
        
        config_path = config_dir / "deployment_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(deployment_config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Deployment config saved: {config_path}")
        
        # Docker config
        docker_config = {
            'base_image': 'pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime',
            'python_version': '3.9',
            'expose_port': 8000,
            'working_dir': '/app',
            'volumes': ['/app/models', '/app/data'],
            'environment': {
                'PYTHONPATH': '/app',
                'MODEL_PATH': '/app/models/best.pt',
                'DEVICE': 'cuda:0'
            }
        }
        
        docker_config_path = config_dir / "docker_config.yaml"
        with open(docker_config_path, 'w') as f:
            yaml.dump(docker_config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Docker config saved: {docker_config_path}")
        
        return deployment_config
    
    def setup_realtime_detection(self):
        """Setup real-time detection system"""
        print("\n🎥 SETTING UP REAL-TIME DETECTION")
        print("-" * 50)
        
        realtime_dir = self.deployment_dir / "realtime"
        realtime_dir.mkdir(parents=True, exist_ok=True)
        
        # Create real-time detection script
        realtime_script = '''#!/usr/bin/env python3
"""
Real-time YOLOv5 Object Detection
================================

This script provides real-time object detection using the trained YOLOv5 model.
Supports camera input, video files, and live streams.
"""

import torch
import cv2
import numpy as np
import time
import argparse
from pathlib import Path
import yaml

class RealTimeDetector:
    def __init__(self, model_path, config_path=None):
        """Initialize real-time detector"""
        self.model_path = model_path
        self.load_config(config_path)
        self.load_model()
        
    def load_config(self, config_path):
        """Load configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'model': {
                    'confidence_threshold': 0.25,
                    'iou_threshold': 0.45,
                    'input_size': [640, 640],
                    'device': 'auto'
                },
                'visualization': {
                    'show_labels': True,
                    'show_confidence': True,
                    'line_thickness': 2,
                    'font_scale': 0.5
                }
            }
    
    def load_model(self):
        """Load YOLOv5 model"""
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
            self.model.conf = self.config['model']['confidence_threshold']
            self.model.iou = self.config['model']['iou_threshold']
            
            # Set device
            device = self.config['model']['device']
            if device == 'auto':
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.model.to(device)
            
            print(f"✅ Model loaded on {device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def detect_camera(self, camera_index=0):
        """Real-time detection from camera"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        
        print(f"🎥 Starting camera detection (Camera {camera_index})")
        print("Press 'q' to quit, 's' to save screenshot")
        
        frame_count = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.model(frame)
            
            # Draw results
            annotated_frame = self.draw_results(frame, results)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
                cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('YOLOv5 Real-time Detection', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = f'screenshot_{int(time.time())}.jpg'
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def detect_video(self, video_path, output_path=None):
        """Process video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎬 Processing video: {video_path}")
        print(f"   Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup output video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.model(frame)
            
            # Draw results
            annotated_frame = self.draw_results(frame, results)
            
            # Save frame if output specified
            if output_path:
                out.write(annotated_frame)
            
            # Show progress
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / frame_count) * (total_frames - frame_count)
                print(f"   Progress: {progress:.1f}% | ETA: {eta:.1f}s")
        
        cap.release()
        if output_path:
            out.release()
            print(f"✅ Output video saved: {output_path}")
    
    def draw_results(self, frame, results):
        """Draw detection results on frame"""
        annotated_frame = frame.copy()
        
        # Get detections
        detections = results.pandas().xyxy[0]
        
        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            confidence = detection['confidence']
            class_name = detection['name']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            if self.config['visualization']['show_labels']:
                label = f"{class_name}"
                if self.config['visualization']['show_confidence']:
                    label += f" {confidence:.2f}"
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_frame

def main():
    parser = argparse.ArgumentParser(description='YOLOv5 Real-time Detection')
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--source', default='0', help='Source: camera index, video file, or image')
    parser.add_argument('--output', help='Output path for video processing')
    parser.add_argument('--config', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = RealTimeDetector(args.model, args.config)
    
    # Determine source type and run detection
    if args.source.isdigit():
        # Camera input
        detector.detect_camera(int(args.source))
    elif Path(args.source).exists():
        # File input
        if args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Video file
            detector.detect_video(args.source, args.output)
        else:
            # Image file
            print("Image detection not implemented in this example")
    else:
        print(f"Invalid source: {args.source}")

if __name__ == "__main__":
    main()
'''
        
        realtime_script_path = realtime_dir / "detect.py"
        with open(realtime_script_path, 'w') as f:
            f.write(realtime_script)
        
        # Make script executable
        os.chmod(realtime_script_path, 0o755)
        
        print(f"✅ Real-time detection script created: {realtime_script_path}")
    
    def create_deployment_packages(self):
        """Create deployment packages"""
        print("\n📦 CREATING DEPLOYMENT PACKAGES")
        print("-" * 50)
        
        packages_dir = self.deployment_dir / "packages"
        
        # 1. Standalone Package
        self.create_standalone_package(packages_dir)
        
        # 2. Docker Package
        self.create_docker_package(packages_dir)
        
        # 3. Cloud API Package
        self.create_cloud_package(packages_dir)
        
        print("✅ All deployment packages created")
    
    def create_standalone_package(self, packages_dir):
        """Create standalone deployment package"""
        standalone_dir = packages_dir / "standalone"
        standalone_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model
        models_dir = standalone_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Copy detection script
        shutil.copy2(
            self.deployment_dir / "realtime" / "detect.py",
            standalone_dir / "detect.py"
        )
        
        # Create setup script
        setup_script = '''#!/bin/bash
# YOLOv5 Standalone Setup Script

echo "🚀 Setting up YOLOv5 Standalone Deployment"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install torch torchvision torchaudio
pip install opencv-python pillow numpy matplotlib
pip install pyyaml pandas seaborn

# Install YOLOv5
pip install ultralytics

echo "✅ Setup completed!"
echo ""
echo "To run detection:"
echo "  source venv/bin/activate"
echo "  python detect.py --model models/best.pt --source 0"
'''
        
        setup_path = standalone_dir / "setup.sh"
        with open(setup_path, 'w') as f:
            f.write(setup_script)
        os.chmod(setup_path, 0o755)
        
        # Create requirements.txt
        requirements = '''torch>=1.12.0
torchvision>=0.13.0
opencv-python>=4.6.0
pillow>=9.0.0
numpy>=1.21.0
matplotlib>=3.5.0
pyyaml>=6.0
pandas>=1.4.0
seaborn>=0.11.0
ultralytics>=8.0.0
'''
        
        with open(standalone_dir / "requirements.txt", 'w') as f:
            f.write(requirements)
        
        print(f"   ✅ Standalone package: {standalone_dir}")
    
    def create_docker_package(self, packages_dir):
        """Create Docker deployment package"""
        docker_dir = packages_dir / "docker"
        docker_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Dockerfile
        dockerfile = '''FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    libglib2.0-0 \\
    libgtk-3-0 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/best.pt

# Run command
CMD ["python", "api.py"]
'''
        
        with open(docker_dir / "Dockerfile", 'w') as f:
            f.write(dockerfile)
        
        # Create docker-compose.yml
        docker_compose = '''version: '3.8'

services:
  yolov5-detector:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - MODEL_PATH=/app/models/best.pt
      - DEVICE=cuda:0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
'''
        
        with open(docker_dir / "docker-compose.yml", 'w') as f:
            f.write(docker_compose)
        
        # Create build and run script
        build_script = '''#!/bin/bash
# Build and run YOLOv5 Docker container

echo "🐳 Building YOLOv5 Docker image..."
docker build -t yolov5-detector .

echo "🚀 Running YOLOv5 container..."
docker run -d \\
  --name yolov5-detector \\
  --gpus all \\
  -p 8000:8000 \\
  -v $(pwd)/models:/app/models \\
  -v $(pwd)/data:/app/data \\
  yolov5-detector

echo "✅ Container started!"
echo "API available at: http://localhost:8000"
echo ""
echo "To stop: docker stop yolov5-detector"
echo "To remove: docker rm yolov5-detector"
'''
        
        build_path = docker_dir / "build_and_run.sh"
        with open(build_path, 'w') as f:
            f.write(build_script)
        os.chmod(build_path, 0o755)
        
        print(f"   ✅ Docker package: {docker_dir}")
    
    def create_cloud_package(self, packages_dir):
        """Create cloud API deployment package"""
        cloud_dir = packages_dir / "cloud"
        cloud_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Flask API
        api_script = '''#!/usr/bin/env python3
"""
YOLOv5 Cloud API
===============

Flask-based REST API for YOLOv5 object detection.
Supports image upload and returns JSON detection results.
"""

from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
import os

app = Flask(__name__)

# Global model variable
model = None

def load_model():
    """Load YOLOv5 model"""
    global model
    model_path = os.getenv('MODEL_PATH', 'models/best.pt')
    
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        model.conf = float(os.getenv('CONFIDENCE_THRESHOLD', '0.25'))
        model.iou = float(os.getenv('IOU_THRESHOLD', '0.45'))
        
        device = os.getenv('DEVICE', 'auto')
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        print(f"✅ Model loaded on {device}")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Object detection endpoint"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get image from request
        image = None
        
        if 'image' in request.files:
            # File upload
            file = request.files['image']
            image = Image.open(file.stream)
        elif 'image_b64' in request.json:
            # Base64 encoded image
            image_data = base64.b64decode(request.json['image_b64'])
            image = Image.open(io.BytesIO(image_data))
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Convert PIL to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run detection
        results = model(image_cv)
        
        # Parse results
        detections = []
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            detections.append({
                'bbox': [float(x) for x in box],
                'confidence': float(conf),
                'class': int(cls),
                'name': model.names[int(cls)]
            })
        
        return jsonify({
            'detections': detections,
            'count': len(detections)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get model classes"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({'classes': model.names})

if __name__ == '__main__':
    print("🚀 Starting YOLOv5 Cloud API...")
    
    if load_model():
        port = int(os.getenv('PORT', '8000'))
        host = os.getenv('HOST', '0.0.0.0')
        debug = os.getenv('DEBUG', 'False').lower() == 'true'
        
        print(f"🌐 API starting on http://{host}:{port}")
        app.run(host=host, port=port, debug=debug)
    else:
        print("❌ Failed to start API - model loading failed")
'''
        
        with open(cloud_dir / "api.py", 'w') as f:
            f.write(api_script)
        
        # Create test client
        test_client = '''#!/usr/bin/env python3
"""
YOLOv5 API Test Client
=====================

Test client for the YOLOv5 Cloud API.
"""

import requests
import base64
import json
import argparse
from pathlib import Path

def test_api(api_url, image_path):
    """Test the API with an image"""
    
    # Health check
    print("🔍 Checking API health...")
    try:
        response = requests.get(f"{api_url}/health")
        print(f"   Status: {response.json()}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return
    
    # Test detection
    print(f"🎯 Testing detection with: {image_path}")
    
    if not Path(image_path).exists():
        print(f"   ❌ Image not found: {image_path}")
        return
    
    try:
        # Method 1: File upload
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{api_url}/detect", files=files)
        
        if response.status_code == 200:
            results = response.json()
            print(f"   ✅ Detection successful!")
            print(f"   📊 Found {results['count']} objects:")
            
            for i, detection in enumerate(results['detections']):
                print(f"      {i+1}. {detection['name']} (confidence: {detection['confidence']:.2f})")
        else:
            print(f"   ❌ Detection failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Request failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test YOLOv5 API')
    parser.add_argument('--url', default='http://localhost:8000', help='API URL')
    parser.add_argument('--image', required=True, help='Path to test image')
    
    args = parser.parse_args()
    test_api(args.url, args.image)

if __name__ == "__main__":
    main()
'''
        
        with open(cloud_dir / "test_client.py", 'w') as f:
            f.write(test_client)
        os.chmod(cloud_dir / "test_client.py", 0o755)
        
        print(f"   ✅ Cloud API package: {cloud_dir}")
    
    def create_documentation(self):
        """Create comprehensive documentation"""
        print("\n📚 CREATING DOCUMENTATION")
        print("-" * 50)
        
        docs_dir = self.deployment_dir / "docs"
        
        # Create README
        self.create_readme()
        
        # Create API documentation
        self.create_api_docs()
        
        # Create troubleshooting guide
        self.create_troubleshooting_guide()
        
        print("✅ Documentation created")
    
    def create_readme(self):
        """Create comprehensive README"""
        readme_content = f'''# YOLOv5 Custom Object Detection - Deployment Guide

## 🎯 Overview

This deployment package contains everything needed to deploy your trained YOLOv5 custom object detection model in various environments.

**Model Information:**
- **Classes**: {len(self.class_names)} ({', '.join(self.class_names)})
- **Framework**: PyTorch/YOLOv5
- **Input Size**: 640×640 pixels
- **Deployment Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📦 Package Contents
