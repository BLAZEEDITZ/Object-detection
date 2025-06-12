"""
YOLOv5 Training Execution
========================

This script handles the actual training process, including:
- Training initiation and monitoring
- Real-time progress tracking
- Performance visualization
- Checkpoint management
"""

import subprocess
import time
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import cv2
from datetime import datetime
import threading
import queue
import os

class YOLOv5Trainer:
    def __init__(self, project_dir):
        """
        Initialize YOLOv5 trainer
        
        Args:
            project_dir: Path to project directory
        """
        self.project_dir = Path(project_dir)
        self.dataset_dir = self.project_dir / "dataset"
        self.configs_dir = self.project_dir / "configs"
        self.results_dir = self.project_dir / "results"
        
        # Load configurations
        self.load_configurations()
        
        # Training state
        self.training_process = None
        self.training_active = False
        self.current_epoch = 0
        self.training_metrics = []
        
        print(f"🚀 YOLOv5 Trainer initialized")
        print(f"📁 Project: {self.project_dir}")
        
    def load_configurations(self):
        """Load training and dataset configurations"""
        # Load dataset config
        dataset_yaml = self.dataset_dir / "dataset.yaml"
        if dataset_yaml.exists():
            with open(dataset_yaml, 'r') as f:
                self.dataset_config = yaml.safe_load(f)
            print(f"✅ Dataset config loaded: {len(self.dataset_config['names'])} classes")
        else:
            raise FileNotFoundError(f"Dataset config not found: {dataset_yaml}")
        
        # Load training config
        training_yaml = self.configs_dir / "optimized_config.yaml"
        if training_yaml.exists():
            with open(training_yaml, 'r') as f:
                self.training_config = yaml.safe_load(f)
            print(f"✅ Training config loaded")
        else:
            print("⚠️  Using default training configuration")
            self.training_config = self.get_default_config()
    
    def get_default_config(self):
        """Get default training configuration"""
        return {
            'model_size': 's',
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640,
            'lr0': 0.01,
            'patience': 30,
            'device': 'auto'
        }
    
    def step5_model_training(self):
        """
        STEP 5: MODEL TRAINING
        =====================
        
        This step covers:
        - Training process initiation
        - Real-time monitoring
        - Progress visualization
        - Checkpoint management
        """
        print("\n" + "="*60)
        print("STEP 5: MODEL TRAINING")
        print("="*60)
        
        print("""
🎯 TRAINING PROCESS OVERVIEW:

1. TRAINING PHASES:
   • Initialization: Model loading and setup
   • Warmup: Learning rate warmup (first few epochs)
   • Main Training: Full training with augmentation
   • Validation: Regular validation checks
   • Early Stopping: Stop if no improvement

2. MONITORING METRICS:
   • Loss Functions: Box, Object, Class losses
   • Accuracy Metrics: Precision, Recall, mAP
   • Learning Rate: Adaptive learning rate changes
   • GPU Utilization: Hardware usage monitoring

3. CHECKPOINTS:
   • best.pt: Best model based on validation mAP
   • last.pt: Most recent model checkpoint
   • Periodic saves: Every N epochs (configurable)

4. REAL-TIME VISUALIZATION:
   • Training curves: Loss and accuracy plots
   • Sample predictions: Visual validation
   • Resource usage: GPU/CPU monitoring
        """)
        
        # Pre-training validation
        self.pre_training_validation()
        
        # Start training
        self.start_training()
        
        # Monitor training
        self.monitor_training()
        
        # Post-training analysis
        self.post_training_analysis()
        
        print("✅ Step 5 Complete: Model Training")
    
    def pre_training_validation(self):
        """Validate everything before starting training"""
        print("\n🔍 PRE-TRAINING VALIDATION...")
        
        # Check YOLOv5 installation
        yolov5_dir = Path("yolov5")
        if not yolov5_dir.exists():
            raise FileNotFoundError("YOLOv5 not found. Please run environment setup first.")
        
        train_script = yolov5_dir / "train.py"
        if not train_script.exists():
            raise FileNotFoundError(f"Training script not found: {train_script}")
        
        # Validate dataset
        dataset_yaml = self.dataset_dir / "dataset.yaml"
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset config not found: {dataset_yaml}")
        
        # Check training data
        train_images = self.dataset_dir / "images" / "train"
        train_labels = self.dataset_dir / "labels" / "train"
        
        if not train_images.exists() or not train_labels.exists():
            raise FileNotFoundError("Training data not found")
        
        # Count training samples
        image_files = list(train_images.glob("*.jpg")) + list(train_images.glob("*.png"))
        label_files = list(train_labels.glob("*.txt"))
        
        print(f"📊 Training data validation:")
        print(f"  Images: {len(image_files)}")
        print(f"  Labels: {len(label_files)}")
        
        if len(image_files) == 0:
            raise ValueError("No training images found")
        
        if len(label_files) == 0:
            raise ValueError("No training labels found")
        
        # Check GPU availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("💻 Training will use CPU (slower)")
        
        print("✅ Pre-training validation passed")
    
    def start_training(self):
        """Start the YOLOv5 training process"""
        print("\n🚀 STARTING TRAINING...")
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"train_{timestamp}"
        self.experiment_dir = self.results_dir / "training_runs" / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Build training command
        yolov5_dir = Path("yolov5")
        train_script = yolov5_dir / "train.py"
        
        cmd = [
            "python", str(train_script),
            "--data", str(self.dataset_dir / "dataset.yaml"),
            "--cfg", f"yolov5{self.training_config['model_size']}.yaml",
            "--weights", f"yolov5{self.training_config['model_size']}.pt",
            "--name", experiment_name,
            "--epochs", str(self.training_config['epochs']),
            "--batch-size", str(self.training_config['batch_size']),
            "--img", str(self.training_config['img_size']),
            "--device", str(self.training_config.get('device', 'auto')),
            "--project", str(self.results_dir / "training_runs"),
            "--exist-ok",
            "--patience", str(self.training_config.get('patience', 30)),
            "--save-period", str(self.training_config.get('save_period', -1))
        ]
        
        # Add hyperparameters
        if 'lr0' in self.training_config:
            cmd.extend(["--hyp", self.create_hyperparameter_file()])
        
        print(f"🔧 Training command:")
        print(" ".join(cmd))
        
        # Start training process
        try:
            # Change to yolov5 directory
            original_dir = os.getcwd()
            os.chdir(yolov5_dir)
            
            print(f"📁 Working directory: {os.getcwd()}")
            print(f"⏰ Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Start training subprocess
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.training_active = True
            print("✅ Training process started successfully")
            
            # Return to original directory
            os.chdir(original_dir)
            
        except Exception as e:
            print(f"❌ Failed to start training: {e}")
            if 'original_dir' in locals():
                os.chdir(original_dir)
            raise
    
    def create_hyperparameter_file(self):
        """Create hyperparameter YAML file"""
        hyp_config = {
            'lr0': self.training_config.get('lr0', 0.01),
            'lrf': self.training_config.get('lrf', 0.01),
            'momentum': self.training_config.get('momentum', 0.937),
            'weight_decay': self.training_config.get('weight_decay', 0.0005),
            'warmup_epochs': self.training_config.get('warmup_epochs', 3),
            'warmup_momentum': self.training_config.get('warmup_momentum', 0.8),
            'warmup_bias_lr': self.training_config.get('warmup_bias_lr', 0.1),
            'box': self.training_config.get('box', 0.05),
            'cls': self.training_config.get('cls', 0.5),
            'obj': self.training_config.get('obj', 1.0),
            'iou_t': self.training_config.get('iou_t', 0.20),
            'anchor_t': self.training_config.get('anchor_t', 4.0),
            'fl_gamma': self.training_config.get('fl_gamma', 0.0),
            'hsv_h': self.training_config.get('hsv_h', 0.015),
            'hsv_s': self.training_config.get('hsv_s', 0.7),
            'hsv_v': self.training_config.get('hsv_v', 0.4),
            'degrees': self.training_config.get('degrees', 0.0),
            'translate': self.training_config.get('translate', 0.1),
            'scale': self.training_config.get('scale', 0.5),
            'shear': self.training_config.get('shear', 0.0),
            'perspective': self.training_config.get('perspective', 0.0),
            'flipud': self.training_config.get('flipud', 0.0),
            'fliplr': self.training_config.get('fliplr', 0.5),
            'mosaic': self.training_config.get('mosaic', 1.0),
            'mixup': self.training_config.get('mixup', 0.0),
            'copy_paste': self.training_config.get('copy_paste', 0.0)
        }
        
        # Save hyperparameter file
        hyp_file = self.experiment_dir / "hyp.yaml"
        with open(hyp_file, 'w') as f:
            yaml.dump(hyp_config, f, default_flow_style=False)
        
        return str(hyp_file)
    
    def monitor_training(self):
        """Monitor training progress in real-time"""
        print("\n📊 MONITORING TRAINING PROGRESS...")
        
        if not self.training_process:
            print("❌ No training process to monitor")
            return
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
        monitor_thread.start()
        
        # Real-time output display
        try:
            while self.training_active:
                output = self.training_process.stdout.readline()
                if output:
                    print(output.strip())
                    self._parse_training_output(output)
                
                # Check if process finished
                if self.training_process.poll() is not None:
                    break
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
            self.stop_training()
        
        # Wait for process to complete
        if self.training_process:
            self.training_process.wait()
            self.training_active = False
        
        print("✅ Training monitoring complete")
    
    def _monitor_worker(self):
        """Background worker for training monitoring"""
        while self.training_active:
            try:
                # Update training visualization every 30 seconds
                if self.current_epoch > 0:
                    self.update_training_plots()
                
                time.sleep(30)
                
            except Exception as e:
                print(f"⚠️  Monitor worker error: {e}")
    
    def _parse_training_output(self, output):
        """Parse training output for metrics"""
        try:
            # Look for epoch information
            if "Epoch" in output and "/" in output:
                # Extract epoch number
                parts = output.split()
                for i, part in enumerate(parts):
                    if "Epoch" in part and i + 1 < len(parts):
                        epoch_info = parts[i + 1]
                        if "/" in epoch_info:
                            current = int(epoch_info.split("/")[0])
                            self.current_epoch = current
                            break
            
            # Look for loss values
            if "train:" in output.lower() and "loss" in output.lower():
                # Parse training metrics
                self._extract_metrics(output)
                
        except Exception as e:
            pass  # Ignore parsing errors
    
    def _extract_metrics(self, output):
        """Extract training metrics from output"""
        try:
            # This is a simplified parser - real implementation would be more robust
            metrics = {}
            
            # Look for common patterns
            if "box_loss" in output:
                # Extract loss values
                parts = output.split()
                for i, part in enumerate(parts):
                    if "box_loss" in part and i + 1 < len(parts):
                        try:
                            metrics['box_loss'] = float(parts[i + 1])
                        except:
                            pass
            
            if metrics:
                metrics['epoch'] = self.current_epoch
                metrics['timestamp'] = time.time()
                self.training_metrics.append(metrics)
                
        except Exception as e:
            pass
    
    def update_training_plots(self):
        """Update training progress plots"""
        if len(self.training_metrics) < 2:
            return
        
        try:
            # Create simple progress plot
            epochs = [m['epoch'] for m in self.training_metrics if 'epoch' in m]
            losses = [m.get('box_loss', 0) for m in self.training_metrics if 'box_loss' in m]
            
            if len(epochs) > 1 and len(losses) > 1:
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, losses, 'b-', label='Box Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Progress')
                plt.legend()
                plt.grid(True)
                
                # Save plot
                plot_path = self.experiment_dir / "training_progress.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"⚠️  Plot update error: {e}")
    
    def stop_training(self):
        """Stop the training process"""
        if self.training_process:
            print("⏹️  Stopping training process...")
            self.training_process.terminate()
            self.training_process.wait()
            self.training_active = False
            print("✅ Training stopped")
    
    def post_training_analysis(self):
        """Analyze training results after completion"""
        print("\n📊 POST-TRAINING ANALYSIS...")
        
        # Find results directory
        results_pattern = f"runs/train/{self.experiment_dir.name}*"
        results_dirs = list(Path("yolov5").glob(results_pattern))
        
        if not results_dirs:
            print("❌ No training results found")
            return
        
        results_dir = results_dirs[0]
        print(f"📁 Results directory: {results_dir}")
        
        # Load training results
        results_file = results_dir / "results.csv"
        if results_file.exists():
            self.analyze_training_results(results_file)
        
        # Check for saved models
        weights_dir = results_dir / "weights"
        if weights_dir.exists():
            best_model = weights_dir / "best.pt"
            last_model = weights_dir / "last.pt"
            
            if best_model.exists():
                print(f"🏆 Best model: {best_model}")
                # Copy to project directory
                project_best = self.project_dir / "models" / "trained" / "best.pt"
                project_best.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(best_model, project_best)
                print(f"📁 Model copied to: {project_best}")
            
            if last_model.exists():
                print(f"💾 Last model: {last_model}")
        
        # Generate training report
        self.generate_training_report(results_dir)
        
        print("✅ Post-training analysis complete")
    
    def analyze_training_results(self, results_file):
        """Analyze training results from CSV file"""
        try:
            df = pd.read_csv(results_file)
            df.columns = df.columns.str.strip()
            
            print(f"📈 TRAINING RESULTS ANALYSIS:")
            
            # Final metrics
            if len(df) > 0:
                final_row = df.iloc[-1]
                
                if 'metrics/mAP_0.5' in df.columns:
                    final_map = final_row['metrics/mAP_0.5']
                    print(f"  Final mAP@0.5: {final_map:.4f}")
                
                if 'metrics/mAP_0.5:0.95' in df.columns:
                    final_map_95 = final_row['metrics/mAP_0.5:0.95']
                    print(f"  Final mAP@0.5:0.95: {final_map_95:.4f}")
                
                if 'metrics/precision' in df.columns:
                    final_precision = final_row['metrics/precision']
                    print(f"  Final Precision: {final_precision:.4f}")
                
                if 'metrics/recall' in df.columns:
                    final_recall = final_row['metrics/recall']
                    print(f"  Final Recall: {final_recall:.4f}")
            
            # Best metrics
            if 'metrics/mAP_0.5' in df.columns:
                best_map = df['metrics/mAP_0.5'].max()
                best_epoch = df.loc[df['metrics/mAP_0.5'].idxmax(), 'epoch']
                print(f"  Best mAP@0.5: {best_map:.4f} (Epoch {best_epoch})")
            
            # Create comprehensive plots
            self.create_training_plots(df)
            
        except Exception as e:
            print(f"❌ Error analyzing results: {e}")
    
    def create_training_plots(self, df):
        """Create comprehensive training plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('YOLOv5 Training Results', fontsize=16)
            
            # Loss plots
            if 'train/box_loss' in df.columns:
                axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss', color='blue')
                if 'train/obj_loss' in df.columns:
                    axes[0, 0].plot(df['epoch'], df['train/obj_loss'], label='Obj Loss', color='red')
                if 'train/cls_loss' in df.columns:
                    axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Cls Loss', color='green')
                
                axes[0, 0].set_title('Training Losses')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # mAP plots
            if 'metrics/mAP_0.5' in df.columns:
                axes[0, 1].plot(df['epoch'], df['metrics/mAP_0.5'], label='mAP@0.5', color='purple')
                if 'metrics/mAP_0.5:0.95' in df.columns:
                    axes[0, 1].plot(df['epoch'], df['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95', color='orange')
                
                axes[0, 1].set_title('Mean Average Precision')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('mAP')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Precision/Recall
            if 'metrics/precision' in df.columns and 'metrics/recall' in df.columns:
                axes[1, 0].plot(df['epoch'], df['metrics/precision'], label='Precision', color='cyan')
                axes[1, 0].plot(df['epoch'], df['metrics/recall'], label='Recall', color='magenta')
                
                axes[1, 0].set_title('Precision & Recall')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Learning rate
            if 'lr/pg0' in df.columns:
                axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='LR pg0', color='brown')
                if 'lr/pg1' in df.columns:
                    axes[1, 1].plot(df['epoch'], df['lr/pg1'], label='LR pg1', color='pink')
                if 'lr/pg2' in df.columns:
                    axes[1, 1].plot(df['epoch'], df['lr/pg2'], label='LR pg2', color='gray')
                
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save plots
            plot_path = self.experiment_dir / "training_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"📈 Training plots saved: {plot_path}")
            
        except Exception as e:
            print(f"⚠️  Error creating plots: {e}")
    
    def generate_training_report(self, results_dir):
        """Generate comprehensive training report"""
        report = {
            'training_config': self.training_config,
            'dataset_config': self.dataset_config,
            'experiment_dir': str(self.experiment_dir),
            'results_dir': str(results_dir),
            'training_start': datetime.now().isoformat(),
            'status': 'completed'
        }
        
        # Add results if available
        results_file = results_dir / "results.csv"
        if results_file.exists():
            try:
                df = pd.read_csv(results_file)
                if len(df) > 0:
                    final_row = df.iloc[-1]
                    report['final_metrics'] = {
                        'epoch': int(final_row.get('epoch', 0)),
                        'mAP_0.5': float(final_row.get('metrics/mAP_0.5', 0)),
                        'mAP_0.5:0.95': float(final_row.get('metrics/mAP_0.5:0.95', 0)),
                        'precision': float(final_row.get('metrics/precision', 0)),
                        'recall': float(final_row.get('metrics/recall', 0))
                    }
                    
                    # Best metrics
                    if 'metrics/mAP_0.5' in df.columns:
                        best_map = df['metrics/mAP_0.5'].max()
                        best_epoch = df.loc[df['metrics/mAP_0.5'].idxmax(), 'epoch']
                        report['best_metrics'] = {
                            'best_mAP_0.5': float(best_map),
                            'best_epoch': int(best_epoch)
                        }
            except Exception as e:
                report['results_error'] = str(e)
        
        # Save report
        report_path = self.experiment_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Training report saved: {report_path}")
        
        return report

def main():
    """Execute YOLOv5 training"""
    print("🚀 YOLOv5 TRAINING EXECUTION")
    print("=" * 60)
    
    # Find project directory
    project_pattern = "yolo_projects/*"
    project_dirs = list(Path(".").glob(project_pattern))
    
    if not project_dirs:
        print("❌ No YOLOv5 projects found")
        print("💡 Please run complete_training_guide.py first")
        return
    
    # Use the most recent project
    project_dir = sorted(project_dirs)[-1]
    print(f"📁 Using project: {project_dir}")
    
    # Initialize trainer
    trainer = YOLOv5Trainer(project_dir)
    
    # Execute training
    try:
        trainer.step5_model_training()
        
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📁 Project: {trainer.project_dir}")
        print(f"🏆 Best model: {trainer.project_dir}/models/trained/best.pt")
        print(f"📊 Results: {trainer.experiment_dir}")
        
        print("\n🚀 Next Steps:")
        print("1. Run performance_evaluation.py to analyze model performance")
        print("2. Run model_optimization.py to optimize for deployment")
        print("3. Run deployment_preparation.py to prepare for real-time use")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Check dataset preparation")
        print("2. Verify YOLOv5 installation")
        print("3. Check available GPU memory")
        print("4. Review training configuration")

if __name__ == "__main__":
    main()
