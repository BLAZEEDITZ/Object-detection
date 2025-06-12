import torch
import torch.nn as nn
import numpy as np
import cv2
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt
from collections import defaultdict

class ModelOptimizer:
    def __init__(self, model_path, dataset_path=None):
        """
        YOLOv5 Model Optimization for Real-time Performance
        
        Args:
            model_path: Path to trained YOLOv5 model
            dataset_path: Path to validation dataset
        """
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path) if dataset_path else None
        
        # Load model
        self.model = self.load_model()
        
        # Optimization results
        self.optimization_results = {}
        
        print(f"🔧 Model Optimizer initialized")
        print(f"📁 Model: {self.model_path}")
        
    def load_model(self):
        """Load YOLOv5 model"""
        try:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(self.model_path))
            print(f"✅ Model loaded successfully")
            return model
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None
    
    def benchmark_model(self, input_sizes=[(640, 640), (416, 416), (320, 320)], num_runs=100):
        """
        Benchmark model performance across different input sizes
        
        Args:
            input_sizes: List of (width, height) tuples to test
            num_runs: Number of inference runs for averaging
        """
        print(f"⚡ Benchmarking model performance...")
        
        benchmark_results = {}
        
        for width, height in input_sizes:
            print(f"📏 Testing size: {width}x{height}")
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, height, width)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = self.model(dummy_input)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                with torch.no_grad():
                    results = self.model(dummy_input)
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            fps = 1.0 / avg_time
            
            benchmark_results[f"{width}x{height}"] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'fps': fps,
                'min_time': np.min(times),
                'max_time': np.max(times)
            }
            
            print(f"  ⏱️  Avg time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
            print(f"  🎬 FPS: {fps:.1f}")
        
        self.optimization_results['benchmark'] = benchmark_results
        return benchmark_results
    
    def optimize_confidence_threshold(self, test_images=None, thresholds=None):
        """
        Optimize confidence threshold for best speed/accuracy tradeoff
        
        Args:
            test_images: List of test image paths
            thresholds: List of confidence thresholds to test
        """
        if thresholds is None:
            thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        
        print(f"🎯 Optimizing confidence threshold...")
        
        if test_images is None:
            # Use sample images from dataset if available
            if self.dataset_path and (self.dataset_path / "images" / "val").exists():
                val_dir = self.dataset_path / "images" / "val"
                test_images = list(val_dir.glob("*.jpg"))[:20]  # Use first 20 images
            else:
                print("⚠️  No test images provided, using synthetic data")
                test_images = self.create_synthetic_test_images()
        
        threshold_results = {}
        
        for threshold in thresholds:
            print(f"🔍 Testing threshold: {threshold}")
            
            # Update model confidence
            self.model.conf = threshold
            
            total_detections = 0
            total_time = 0
            detection_counts = []
            
            for img_path in test_images:
                start_time = time.time()
                
                if isinstance(img_path, str) or isinstance(img_path, Path):
                    results = self.model(str(img_path))
                else:
                    results = self.model(img_path)
                
                end_time = time.time()
                
                # Count detections
                detections = len(results.pandas().xyxy[0])
                total_detections += detections
                detection_counts.append(detections)
                total_time += (end_time - start_time)
            
            avg_detections = np.mean(detection_counts)
            avg_time = total_time / len(test_images)
            fps = 1.0 / avg_time
            
            threshold_results[threshold] = {
                'avg_detections': avg_detections,
                'total_detections': total_detections,
                'avg_time': avg_time,
                'fps': fps,
                'detection_variance': np.var(detection_counts)
            }
            
            print(f"  📊 Avg detections: {avg_detections:.1f}")
            print(f"  ⚡ FPS: {fps:.1f}")
        
        # Find optimal threshold (balance between detections and speed)
        optimal_threshold = self.find_optimal_threshold(threshold_results)
        
        self.optimization_results['confidence_threshold'] = {
            'results': threshold_results,
            'optimal': optimal_threshold
        }
        
        print(f"🏆 Optimal confidence threshold: {optimal_threshold}")
        return threshold_results, optimal_threshold
    
    def find_optimal_threshold(self, threshold_results):
        """Find optimal confidence threshold based on detection/speed tradeoff"""
        scores = {}
        
        # Normalize metrics
        fps_values = [r['fps'] for r in threshold_results.values()]
        detection_values = [r['avg_detections'] for r in threshold_results.values()]
        
        max_fps = max(fps_values)
        max_detections = max(detection_values)
        
        for threshold, results in threshold_results.items():
            # Weighted score: 60% speed, 40% detections
            fps_score = results['fps'] / max_fps
            detection_score = results['avg_detections'] / max_detections if max_detections > 0 else 0
            
            combined_score = 0.6 * fps_score + 0.4 * detection_score
            scores[threshold] = combined_score
        
        return max(scores, key=scores.get)
    
    def optimize_nms_threshold(self, test_images=None, iou_thresholds=None):
        """
        Optimize Non-Maximum Suppression (NMS) IoU threshold
        
        Args:
            test_images: List of test image paths
            iou_thresholds: List of IoU thresholds to test
        """
        if iou_thresholds is None:
            iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7]
        
        print(f"🔧 Optimizing NMS IoU threshold...")
        
        if test_images is None:
            if self.dataset_path and (self.dataset_path / "images" / "val").exists():
                val_dir = self.dataset_path / "images" / "val"
                test_images = list(val_dir.glob("*.jpg"))[:15]
            else:
                test_images = self.create_synthetic_test_images()
        
        nms_results = {}
        
        for iou_threshold in iou_thresholds:
            print(f"🎯 Testing IoU threshold: {iou_threshold}")
            
            # Update model IoU threshold
            self.model.iou = iou_threshold
            
            total_detections = 0
            total_time = 0
            
            for img_path in test_images:
                start_time = time.time()
                results = self.model(str(img_path) if isinstance(img_path, Path) else img_path)
                end_time = time.time()
                
                detections = len(results.pandas().xyxy[0])
                total_detections += detections
                total_time += (end_time - start_time)
            
            avg_detections = total_detections / len(test_images)
            avg_time = total_time / len(test_images)
            fps = 1.0 / avg_time
            
            nms_results[iou_threshold] = {
                'avg_detections': avg_detections,
                'avg_time': avg_time,
                'fps': fps
            }
            
            print(f"  📊 Avg detections: {avg_detections:.1f}")
            print(f"  ⚡ FPS: {fps:.1f}")
        
        self.optimization_results['nms_threshold'] = nms_results
        return nms_results
    
    def create_synthetic_test_images(self, num_images=10):
        """Create synthetic test images for optimization"""
        synthetic_images = []
        
        for i in range(num_images):
            # Create random image
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add some objects
            cv2.rectangle(image, (100 + i*20, 100), (200 + i*20, 200), (255, 0, 0), -1)
            cv2.circle(image, (400, 300 + i*10), 50, (0, 255, 0), -1)
            
            synthetic_images.append(image)
        
        return synthetic_images
    
    def export_optimized_model(self, output_format='torchscript', output_path=None):
        """
        Export model in optimized format
        
        Args:
            output_format: Export format ('torchscript', 'onnx', 'tensorrt')
            output_path: Output file path
        """
        if output_path is None:
            output_path = self.model_path.parent / f"optimized_model.{output_format}"
        
        print(f"📦 Exporting model to {output_format} format...")
        
        try:
            if output_format.lower() == 'torchscript':
                # Export to TorchScript
                dummy_input = torch.randn(1, 3, 640, 640)
                traced_model = torch.jit.trace(self.model.model, dummy_input)
                traced_model.save(str(output_path))
                
            elif output_format.lower() == 'onnx':
                # Export to ONNX
                dummy_input = torch.randn(1, 3, 640, 640)
                torch.onnx.export(
                    self.model.model,
                    dummy_input,
                    str(output_path),
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output']
                )
                
            else:
                print(f"❌ Unsupported export format: {output_format}")
                return False
            
            print(f"✅ Model exported successfully: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False
    
    def create_optimization_report(self):
        """Create comprehensive optimization report"""
        print("📊 Creating optimization report...")
        
        report = {
            'model_path': str(self.model_path),
            'optimization_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': self.optimization_results
        }
        
        # Save report
        report_path = self.model_path.parent / "optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create visualization
        self.plot_optimization_results()
        
        print(f"📄 Optimization report saved: {report_path}")
        return report
    
    def plot_optimization_results(self):
        """Plot optimization results"""
        if not self.optimization_results:
            print("⚠️  No optimization results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('YOLOv5 Model Optimization Results', fontsize=16)
        
        # Benchmark results
        if 'benchmark' in self.optimization_results:
            benchmark_data = self.optimization_results['benchmark']
            sizes = list(benchmark_data.keys())
            fps_values = [data['fps'] for data in benchmark_data.values()]
            
            axes[0, 0].bar(sizes, fps_values, color='skyblue')
            axes[0, 0].set_title('FPS vs Input Size')
            axes[0, 0].set_xlabel('Input Size')
            axes[0, 0].set_ylabel('FPS')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Confidence threshold optimization
        if 'confidence_threshold' in self.optimization_results:
            conf_data = self.optimization_results['confidence_threshold']['results']
            thresholds = list(conf_data.keys())
            fps_values = [data['fps'] for data in conf_data.values()]
            detection_values = [data['avg_detections'] for data in conf_data.values()]
            
            ax1 = axes[0, 1]
            ax2 = ax1.twinx()
            
            line1 = ax1.plot(thresholds, fps_values, 'b-', label='FPS')
            line2 = ax2.plot(thresholds, detection_values, 'r-', label='Avg Detections')
            
            ax1.set_xlabel('Confidence Threshold')
            ax1.set_ylabel('FPS', color='b')
            ax2.set_ylabel('Avg Detections', color='r')
            ax1.set_title('Confidence Threshold Optimization')
            
            # Add optimal threshold line
            if 'optimal' in self.optimization_results['confidence_threshold']:
                optimal = self.optimization_results['confidence_threshold']['optimal']
                ax1.axvline(x=optimal, color='g', linestyle='--', label=f'Optimal: {optimal}')
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
        
        # NMS threshold optimization
        if 'nms_threshold' in self.optimization_results:
            nms_data = self.optimization_results['nms_threshold']
            iou_thresholds = list(nms_data.keys())
            fps_values = [data['fps'] for data in nms_data.values()]
            detection_values = [data['avg_detections'] for data in nms_data.values()]
            
            ax1 = axes[1, 0]
            ax2 = ax1.twinx()
            
            line1 = ax1.plot(iou_thresholds, fps_values, 'b-', label='FPS')
            line2 = ax2.plot(iou_thresholds, detection_values, 'r-', label='Avg Detections')
            
            ax1.set_xlabel('IoU Threshold')
            ax1.set_ylabel('FPS', color='b')
            ax2.set_ylabel('Avg Detections', color='r')
            ax1.set_title('NMS IoU Threshold Optimization')
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
        
        # Summary statistics
        axes[1, 1].axis('off')
        summary_text = "Optimization Summary:\n\n"
        
        if 'benchmark' in self.optimization_results:
            best_size = max(self.optimization_results['benchmark'].items(), 
                          key=lambda x: x[1]['fps'])
            summary_text += f"Best Input Size: {best_size[0]}\n"
            summary_text += f"Max FPS: {best_size[1]['fps']:.1f}\n\n"
        
        if 'confidence_threshold' in self.optimization_results:
            optimal_conf = self.optimization_results['confidence_threshold']['optimal']
            summary_text += f"Optimal Confidence: {optimal_conf}\n\n"
        
        summary_text += "Recommendations:\n"
        summary_text += "• Use smaller input sizes for speed\n"
        summary_text += "• Adjust confidence threshold based on use case\n"
        summary_text += "• Consider model quantization for mobile deployment"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.model_path.parent / "optimization_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Optimization plots saved: {plot_path}")

def main():
    """Demonstrate model optimization"""
    print("🚀 YOLOv5 Model Optimization")
    print("=" * 60)
    
    # Example model path (adjust as needed)
    model_path = "runs/train/custom_yolov5s_*/weights/best.pt"
    dataset_path = "datasets/custom_objects"
    
    # Check if model exists
    model_files = list(Path(".").glob(model_path))
    if not model_files:
        print(f"❌ No trained model found matching: {model_path}")
        print("💡 Please train a model first using training_pipeline.py")
        return
    
    # Use the most recent model
    model_file = sorted(model_files)[-1]
    print(f"📁 Using model: {model_file}")
    
    # Initialize optimizer
    optimizer = ModelOptimizer(model_file, dataset_path)
    
    if optimizer.model is None:
        print("❌ Failed to load model")
        return
    
    # Run optimizations
    print("\n⚡ Running performance benchmark...")
    benchmark_results = optimizer.benchmark_model()
    
    print("\n🎯 Optimizing confidence threshold...")
    conf_results, optimal_conf = optimizer.optimize_confidence_threshold()
    
    print("\n🔧 Optimizing NMS threshold...")
    nms_results = optimizer.optimize_nms_threshold()
    
    # Create optimization report
    print("\n📊 Creating optimization report...")
    report = optimizer.create_optimization_report()
    
    # Export optimized model
    print("\n📦 Exporting optimized model...")
    optimizer.export_optimized_model('torchscript')
    
    print("\n✅ Model optimization complete!")
    print(f"🏆 Optimal confidence threshold: {optimal_conf}")
    print("💡 Check the optimization report and plots for detailed results")

if __name__ == "__main__":
    main()
