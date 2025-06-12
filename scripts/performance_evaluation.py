"""
YOLOv5 Performance Evaluation
============================

This script provides comprehensive model evaluation including:
- Validation metrics analysis
- Test set evaluation
- Confusion matrix generation
- Per-class performance analysis
- Visual result inspection
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
import yaml

class YOLOv5Evaluator:
    def __init__(self, project_dir):
        """
        Initialize YOLOv5 model evaluator
        
        Args:
            project_dir: Path to project directory
        """
        self.project_dir = Path(project_dir)
        self.dataset_dir = self.project_dir / "dataset"
        self.models_dir = self.project_dir / "models" / "trained"
        self.results_dir = self.project_dir / "results" / "evaluations"
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        self.load_configurations()
        
        # Load trained model
        self.model = self.load_trained_model()
        
        print(f"📊 YOLOv5 Evaluator initialized")
        print(f"📁 Project: {self.project_dir}")
    
    def load_configurations(self):
        """Load dataset and class configurations"""
        # Load dataset config
        dataset_yaml = self.dataset_dir / "dataset.yaml"
        if dataset_yaml.exists():
            with open(dataset_yaml, 'r') as f:
                self.dataset_config = yaml.safe_load(f)
            self.class_names = self.dataset_config['names']
            print(f"✅ Dataset config loaded: {len(self.class_names)} classes")
        else:
            raise FileNotFoundError(f"Dataset config not found: {dataset_yaml}")
    
    def load_trained_model(self):
        """Load the trained YOLOv5 model"""
        model_path = self.models_dir / "best.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        
        try:
            # Load custom trained model
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path))
            print(f"✅ Trained model loaded: {model_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def step6_performance_evaluation(self):
        """
        STEP 6: PERFORMANCE EVALUATION
        ==============================
        
        This step covers:
        - Validation metrics analysis
        - Test set evaluation
        - Confusion matrix generation
        - Per-class performance analysis
        - Error analysis and visualization
        """
        print("\n" + "="*60)
        print("STEP 6: PERFORMANCE EVALUATION")
        print("="*60)
        
        print("""
📊 EVALUATION METRICS OVERVIEW:

1. DETECTION METRICS:
   • mAP@0.5: Mean Average Precision at IoU threshold 0.5
   • mAP@0.5:0.95: Mean Average Precision averaged over IoU 0.5-0.95
   • Precision: True Positives / (True Positives + False Positives)
   • Recall: True Positives / (True Positives + False Negatives)

2. PER-CLASS ANALYSIS:
   • Individual class performance
   • Class-specific precision and recall
   • Confusion matrix analysis
   • Difficult cases identification

3. VISUAL EVALUATION:
   • Sample predictions on test set
   • Error case analysis
   • Confidence distribution
   • Bounding box quality assessment

4. PERFORMANCE INSIGHTS:
   • Model strengths and weaknesses
   • Recommendations for improvement
   • Dataset quality assessment
        """)
        
        # Evaluate on validation set
        self.evaluate_validation_set()
        
        # Evaluate on test set
        self.evaluate_test_set()
        
        # Generate confusion matrix
        self.generate_confusion_matrix()
        
        # Per-class analysis
        self.analyze_per_class_performance()
        
        # Visual evaluation
        self.visual_evaluation()
        
        # Generate comprehensive report
        self.generate_evaluation_report()
        
        print("✅ Step 6 Complete: Performance Evaluation")
    
    def evaluate_validation_set(self):
        """Evaluate model on validation set"""
        print("\n📊 EVALUATING ON VALIDATION SET...")
        
        val_images_dir = self.dataset_dir / "images" / "val"
        val_labels_dir = self.dataset_dir / "labels" / "val"
        
        if not val_images_dir.exists():
            print("⚠️  No validation set found")
            return
        
        # Get validation images
        image_files = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png"))
        
        if not image_files:
            print("⚠️  No validation images found")
            return
        
        print(f"🖼️  Processing {len(image_files)} validation images...")
        
        # Evaluation metrics
        total_predictions = 0
        total_ground_truth = 0
        correct_predictions = 0
        class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0})
        
        # Process each image
        for image_file in image_files:
            # Load image
            image = cv2.imread(str(image_file))
            if image is None:
                continue
            
            # Get predictions
            results = self.model(image)
            predictions = results.pandas().xyxy[0]
            
            # Load ground truth
            label_file = val_labels_dir / f"{image_file.stem}.txt"
            ground_truth = self.load_ground_truth(label_file, image.shape)
            
            # Calculate metrics for this image
            image_metrics = self.calculate_image_metrics(predictions, ground_truth)
            
            # Update totals
            total_predictions += len(predictions)
            total_ground_truth += len(ground_truth)
            
            # Update class-specific metrics
            for class_id in range(len(self.class_names)):
                class_metrics[class_id]['tp'] += image_metrics['class_tp'][class_id]
                class_metrics[class_id]['fp'] += image_metrics['class_fp'][class_id]
                class_metrics[class_id]['fn'] += image_metrics['class_fn'][class_id]
                class_metrics[class_id]['total_gt'] += image_metrics['class_gt'][class_id]
        
        # Calculate overall metrics
        overall_precision = sum(class_metrics[c]['tp'] for c in class_metrics) / max(1, total_predictions)
        overall_recall = sum(class_metrics[c]['tp'] for c in class_metrics) / max(1, total_ground_truth)
        overall_f1 = 2 * (overall_precision * overall_recall) / max(1e-6, overall_precision + overall_recall)
        
        # Calculate per-class metrics
        class_results = {}
        for class_id, metrics in class_metrics.items():
            if class_id < len(self.class_names):
                tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                f1 = 2 * (precision * recall) / max(1e-6, precision + recall)
                
                class_results[self.class_names[class_id]] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': metrics['total_gt']
                }
        
        # Print results
        print(f"📊 VALIDATION SET RESULTS:")
        print(f"  Overall Precision: {overall_precision:.4f}")
        print(f"  Overall Recall: {overall_recall:.4f}")
        print(f"  Overall F1-Score: {overall_f1:.4f}")
        print(f"  Total Predictions: {total_predictions}")
        print(f"  Total Ground Truth: {total_ground_truth}")
        
        # Save validation results
        validation_results = {
            'overall_metrics': {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1_score': overall_f1,
                'total_predictions': total_predictions,
                'total_ground_truth': total_ground_truth
            },
            'class_metrics': class_results
        }
        
        results_file = self.results_dir / "validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        print(f"📄 Validation results saved: {results_file}")
        
        return validation_results
    
    def load_ground_truth(self, label_file, image_shape):
        """Load ground truth annotations"""
        ground_truth = []
        
        if not label_file.exists():
            return ground_truth
        
        height, width = image_shape[:2]
        
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    box_width = float(parts[3]) * width
                    box_height = float(parts[4]) * height
                    
                    x1 = x_center - box_width / 2
                    y1 = y_center - box_height / 2
                    x2 = x_center + box_width / 2
                    y2 = y_center + box_height / 2
                    
                    ground_truth.append({
                        'class_id': class_id,
                        'bbox': [x1, y1, x2, y2],
                        'matched': False
                    })
        
        except Exception as e:
            print(f"⚠️  Error loading ground truth from {label_file}: {e}")
        
        return ground_truth
    
    def calculate_image_metrics(self, predictions, ground_truth, iou_threshold=0.5):
        """Calculate metrics for a single image"""
        metrics = {
            'class_tp': [0] * len(self.class_names),
            'class_fp': [0] * len(self.class_names),
            'class_fn': [0] * len(self.class_names),
            'class_gt': [0] * len(self.class_names)
        }
        
        # Count ground truth per class
        for gt in ground_truth:
            if gt['class_id'] < len(self.class_names):
                metrics['class_gt'][gt['class_id']] += 1
        
        # Match predictions to ground truth
        for _, pred in predictions.iterrows():
            pred_class = self.class_names.index(pred['name']) if pred['name'] in self.class_names else -1
            if pred_class == -1:
                continue
            
            pred_bbox = [pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax']]
            
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt['class_id'] == pred_class and not gt['matched']:
                    iou = self.calculate_iou(pred_bbox, gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            # Determine if prediction is correct
            if best_iou >= iou_threshold:
                metrics['class_tp'][pred_class] += 1
                ground_truth[best_gt_idx]['matched'] = True
            else:
                metrics['class_fp'][pred_class] += 1
        
        # Count false negatives (unmatched ground truth)
        for gt in ground_truth:
            if not gt['matched'] and gt['class_id'] < len(self.class_names):
                metrics['class_fn'][gt['class_id']] += 1
        
        return metrics
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-6)
    
    def evaluate_test_set(self):
        """Evaluate model on test set"""
        print("\n🧪 EVALUATING ON TEST SET...")
        
        test_images_dir = self.dataset_dir / "images" / "test"
        
        if not test_images_dir.exists():
            print("⚠️  No test set found")
            return
        
        image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
        
        if not image_files:
            print("⚠️  No test images found")
            return
        
        print(f"🖼️  Processing {len(image_files)} test images...")
        
        # Test set evaluation
        test_results = {
            'total_images': len(image_files),
            'total_detections': 0,
            'confidence_distribution': [],
            'class_distribution': defaultdict(int),
            'detection_sizes': [],
            'processing_times': []
        }
        
        # Process test images
        for image_file in image_files:
            image = cv2.imread(str(image_file))
            if image is None:
                continue
            
            # Time the inference
            import time
            start_time = time.time()
            results = self.model(image)
            end_time = time.time()
            
            processing_time = end_time - start_time
            test_results['processing_times'].append(processing_time)
            
            # Analyze predictions
            predictions = results.pandas().xyxy[0]
            test_results['total_detections'] += len(predictions)
            
            for _, pred in predictions.iterrows():
                # Confidence distribution
                test_results['confidence_distribution'].append(pred['confidence'])
                
                # Class distribution
                test_results['class_distribution'][pred['name']] += 1
                
                # Detection size
                width = pred['xmax'] - pred['xmin']
                height = pred['ymax'] - pred['ymin']
                area = width * height
                test_results['detection_sizes'].append(area)
        
        # Calculate statistics
        if test_results['processing_times']:
            avg_processing_time = np.mean(test_results['processing_times'])
            fps = 1.0 / avg_processing_time
            test_results['avg_processing_time'] = avg_processing_time
            test_results['fps'] = fps
        
        if test_results['confidence_distribution']:
            test_results['avg_confidence'] = np.mean(test_results['confidence_distribution'])
            test_results['confidence_std'] = np.std(test_results['confidence_distribution'])
        
        # Print test results
        print(f"📊 TEST SET RESULTS:")
        print(f"  Total Images: {test_results['total_images']}")
        print(f"  Total Detections: {test_results['total_detections']}")
        print(f"  Avg Detections per Image: {test_results['total_detections']/test_results['total_images']:.2f}")
        
        if 'avg_processing_time' in test_results:
            print(f"  Avg Processing Time: {test_results['avg_processing_time']*1000:.1f}ms")
            print(f"  FPS: {test_results['fps']:.1f}")
        
        if 'avg_confidence' in test_results:
            print(f"  Avg Confidence: {test_results['avg_confidence']:.3f}")
        
        # Save test results
        results_file = self.results_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"📄 Test results saved: {results_file}")
        
        return test_results
    
    def generate_confusion_matrix(self):
        """Generate confusion matrix for model predictions"""
        print("\n📊 GENERATING CONFUSION MATRIX...")
        
        val_images_dir = self.dataset_dir / "images" / "val"
        val_labels_dir = self.dataset_dir / "labels" / "val"
        
        if not val_images_dir.exists():
            print("⚠️  No validation set for confusion matrix")
            return
        
        # Collect predictions and ground truth
        y_true = []
        y_pred = []
        
        image_files = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png"))
        
        for image_file in image_files[:50]:  # Limit for demo
            image = cv2.imread(str(image_file))
            if image is None:
                continue
            
            # Get predictions
            results = self.model(image)
            predictions = results.pandas().xyxy[0]
            
            # Load ground truth
            label_file = val_labels_dir / f"{image_file.stem}.txt"
            ground_truth = self.load_ground_truth(label_file, image.shape)
            
            # Match predictions to ground truth for confusion matrix
            for gt in ground_truth:
                if gt['class_id'] < len(self.class_names):
                    y_true.append(gt['class_id'])
                    
                    # Find best matching prediction
                    best_match = None
                    best_iou = 0
                    
                    for _, pred in predictions.iterrows():
                        pred_class = self.class_names.index(pred['name']) if pred['name'] in self.class_names else -1
                        if pred_class != -1:
                            pred_bbox = [pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax']]
                            iou = self.calculate_iou(pred_bbox, gt['bbox'])
                            if iou > best_iou:
                                best_iou = iou
                                best_match = pred_class
                    
                    if best_match is not None and best_iou > 0.5:
                        y_pred.append(best_match)
                    else:
                        y_pred.append(-1)  # No match found
        
        if len(y_true) == 0:
            print("⚠️  No ground truth data for confusion matrix")
            return
        
        # Create confusion matrix
        class_labels = list(range(len(self.class_names))) + [-1]
        class_names_extended = self.class_names + ['No Detection']
        
        cm = confusion_matrix(y_true, y_pred, labels=class_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names_extended,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Save confusion matrix
        cm_path = self.results_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Confusion matrix saved: {cm_path}")
        
        return cm
    
    def analyze_per_class_performance(self):
        """Analyze performance for each class"""
        print("\n📈 ANALYZING PER-CLASS PERFORMANCE...")
        
        # Load validation results
        val_results_file = self.results_dir / "validation_results.json"
        if not val_results_file.exists():
            print("⚠️  No validation results found")
            return
        
        with open(val_results_file, 'r') as f:
            val_results = json.load(f)
        
        class_metrics = val_results.get('class_metrics', {})
        
        if not class_metrics:
            print("⚠️  No class metrics found")
            return
        
        # Create per-class performance plot
        classes = list(class_metrics.keys())
        precisions = [class_metrics[cls]['precision'] for cls in classes]
        recalls = [class_metrics[cls]['recall'] for cls in classes]
        f1_scores = [class_metrics[cls]['f1'] for cls in classes]
        supports = [class_metrics[cls]['support'] for cls in classes]
        
        # Create subplot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Per-Class Performance Analysis', fontsize=16)
        
        # Precision by class
        axes[0, 0].bar(classes, precisions, color='skyblue')
        axes[0, 0].set_title('Precision by Class')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Recall by class
        axes[0, 1].bar(classes, recalls, color='lightgreen')
        axes[0, 1].set_title('Recall by Class')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1-Score by class
        axes[1, 0].bar(classes, f1_scores, color='lightcoral')
        axes[1, 0].set_title('F1-Score by Class')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Support (number of samples) by class
        axes[1, 1].bar(classes, supports, color='gold')
        axes[1, 1].set_title('Support by Class')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / "per_class_performance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed analysis
        print(f"📊 PER-CLASS PERFORMANCE ANALYSIS:")
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        
        for cls in classes:
            metrics = class_metrics[cls]
            print(f"{cls:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                  f"{metrics['f1']:<10.3f} {metrics['support']:<10}")
        
        # Identify best and worst performing classes
        best_f1_class = max(classes, key=lambda x: class_metrics[x]['f1'])
        worst_f1_class = min(classes, key=lambda x: class_metrics[x]['f1'])
        
        print(f"\n🏆 Best performing class: {best_f1_class} (F1: {class_metrics[best_f1_class]['f1']:.3f})")
        print(f"⚠️  Worst performing class: {worst_f1_class} (F1: {class_metrics[worst_f1_class]['f1']:.3f})")
        
        print(f"📈 Performance plot saved: {plot_path}")
    
    def visual_evaluation(self):
        """Perform visual evaluation of model predictions"""
        print("\n👁️  PERFORMING VISUAL EVALUATION...")
        
        test_images_dir = self.dataset_dir / "images" / "test"
        
        if not test_images_dir.exists():
            test_images_dir = self.dataset_dir / "images" / "val"
        
        if not test_images_dir.exists():
            print("⚠️  No images found for visual evaluation")
            return
        
        image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
        
        if not image_files:
            print("⚠️  No image files found")
            return
        
        # Select random sample of images
        sample_size = min(12, len(image_files))
        sample_images = np.random.choice(image_files, sample_size, replace=False)
        
        # Create visualization
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Model Predictions - Visual Evaluation', fontsize=16)
        
        for idx, image_file in enumerate(sample_images):
            row = idx // 4
            col = idx % 4
            
            # Load and process image
            image = cv2.imread(str(image_file))
            if image is None:
                continue
            
            # Get predictions
            results = self.model(image)
            
            # Draw predictions
            annotated_image = self.draw_predictions(image, results)
            
            # Convert BGR to RGB for matplotlib
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            # Display
            axes[row, col].imshow(annotated_image)
            axes[row, col].set_title(f"{image_file.name}", fontsize=10)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.results_dir / "visual_evaluation.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"👁️  Visual evaluation saved: {viz_path}")
    
    def draw_predictions(self, image, results):
        """Draw predictions on image"""
        annotated_image = image.copy()
        predictions = results.pandas().xyxy[0]
        
        # Color map for classes
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
        
        for _, pred in predictions.iterrows():
            x1, y1, x2, y2 = int(pred['xmin']), int(pred['ymin']), int(pred['xmax']), int(pred['ymax'])
            confidence = pred['confidence']
            class_name = pred['name']
            
            # Get color for this class
            class_id = self.class_names.index(class_name) if class_name in self.class_names else 0
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_image
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        print("\n📄 GENERATING EVALUATION REPORT...")
        
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_path': str(self.models_dir / "best.pt"),
            'dataset_config': self.dataset_config,
            'class_names': self.class_names
        }
        
        # Load validation results
        val_results_file = self.results_dir / "validation_results.json"
        if val_results_file.exists():
            with open(val_results_file, 'r') as f:
                report['validation_results'] = json.load(f)
        
        # Load test results
        test_results_file = self.results_dir / "test_results.json"
        if test_results_file.exists():
            with open(test_results_file, 'r') as f:
                report['test_results'] = json.load(f)
        
        # Add recommendations
        report['recommendations'] = self.generate_recommendations(report)
        
        # Save comprehensive report
        report_path = self.results_dir / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create summary report
        self.create_summary_report(report)
        
        print(f"📄 Evaluation report saved: {report_path}")
        
        return report
    
    def generate_recommendations(self, report):
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Check validation results
        if 'validation_results' in report:
            val_results = report['validation_results']
            overall_metrics = val_results.get('overall_metrics', {})
            
            precision = overall_metrics.get('precision', 0)
            recall = overall_metrics.get('recall', 0)
            f1_score = overall_metrics.get('f1_score', 0)
            
            if f1_score < 0.5:
                recommendations.append("Low F1-score detected. Consider collecting more training data or adjusting model architecture.")
            
            if precision < 0.6:
                recommendations.append("Low precision detected. Consider increasing confidence threshold or improving annotation quality.")
            
            if recall < 0.6:
                recommendations.append("Low recall detected. Consider lowering confidence threshold or adding more diverse training examples.")
            
            # Check class balance
            class_metrics = val_results.get('class_metrics', {})
            if class_metrics:
                f1_scores = [metrics['f1'] for metrics in class_metrics.values()]
                if max(f1_scores) - min(f1_scores) > 0.3:
                    recommendations.append("High variance in class performance. Consider balancing dataset or using class weights.")
        
        # Check test results
        if 'test_results' in report:
            test_results = report['test_results']
            
            fps = test_results.get('fps', 0)
            if fps < 10:
                recommendations.append("Low inference speed. Consider model optimization or using a smaller model variant.")
            
            avg_confidence = test_results.get('avg_confidence', 0)
            if avg_confidence < 0.7:
                recommendations.append("Low average confidence. Model may need more training or better data quality.")
        
        if not recommendations:
            recommendations.append("Model performance looks good! Consider fine-tuning for specific deployment requirements.")
        
        return recommendations
    
    def create_summary_report(self, report):
        """Create human-readable summary report"""
        summary_lines = []
        summary_lines.append("YOLOv5 MODEL EVALUATION SUMMARY")
        summary_lines.append("=" * 50)
        summary_lines.append("")
        
        # Model info
        summary_lines.append(f"Model: {report['model_path']}")
        summary_lines.append(f"Classes: {len(report['class_names'])} - {', '.join(report['class_names'])}")
        summary_lines.append(f"Evaluation Date: {report['evaluation_timestamp']}")
        summary_lines.append("")
        
        # Validation results
        if 'validation_results' in report:
            val_results = report['validation_results']
            overall = val_results.get('overall_metrics', {})
            
            summary_lines.append("VALIDATION SET PERFORMANCE:")
            summary_lines.append(f"  Precision: {overall.get('precision', 0):.4f}")
            summary_lines.append(f"  Recall: {overall.get('recall', 0):.4f}")
            summary_lines.append(f"  F1-Score: {overall.get('f1_score', 0):.4f}")
            summary_lines.append("")
        
        # Test results
        if 'test_results' in report:
            test_results = report['test_results']
            
            summary_lines.append("TEST SET PERFORMANCE:")
            summary_lines.append(f"  Processing Speed: {test_results.get('fps', 0):.1f} FPS")
            summary_lines.append(f"  Average Confidence: {test_results.get('avg_confidence', 0):.3f}")
            summary_lines.append(f"  Total Detections: {test_results.get('total_detections', 0)}")
            summary_lines.append("")
        
        # Recommendations
        if 'recommendations' in report:
            summary_lines.append("RECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                summary_lines.append(f"  {i}. {rec}")
            summary_lines.append("")
        
        # Save summary
        summary_path = self.results_dir / "evaluation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        # Print summary
        print('\n'.join(summary_lines))
        print(f"📄 Summary report saved: {summary_path}")

def main():
    """Execute YOLOv5 performance evaluation"""
    print("🚀 YOLOv5 PERFORMANCE EVALUATION")
    print("=" * 60)
    
    # Find project directory
    project_pattern = "yolo_projects/*"
    project_dirs = list(Path(".").glob(project_pattern))
    
    if not project_dirs:
        print("❌ No YOLOv5 projects found")
        print("💡 Please run training first")
        return
    
    # Use the most recent project
    project_dir = sorted(project_dirs)[-1]
    print(f"📁 Using project: {project_dir}")
    
    # Check if model exists
    model_path = project_dir / "models" / "trained" / "best.pt"
    if not model_path.exists():
        print("❌ No trained model found")
        print("💡 Please complete training first")
        return
    
    # Initialize evaluator
    try:
        evaluator = YOLOv5Evaluator(project_dir)
        
        # Execute evaluation
        evaluator.step6_performance_evaluation()
        
        print("\n🎉 EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📁 Project: {evaluator.project_dir}")
        print(f"📊 Results: {evaluator.results_dir}")
        
        print("\n🚀 Next Steps:")
        print("1. Review evaluation results and recommendations")
        print("2. Run model_optimization.py to optimize for deployment")
        print("3. Run deployment_preparation.py to prepare for real-time use")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Ensure model training completed successfully")
        print("2. Check dataset structure and annotations")
        print("3. Verify YOLOv5 installation")

if __name__ == "__main__":
    main()
