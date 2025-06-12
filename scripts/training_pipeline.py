import torch
import yaml
import os
import shutil
from pathlib import Path
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import wandb  # For experiment tracking (optional)

class YOLOv5Trainer:
    def __init__(self, dataset_path, model_size='s', device='auto'):
        """
        YOLOv5 Custom Training Pipeline
        
        Args:
            dataset_path: Path to dataset directory
            model_size: Model size ('s', 'm', 'l', 'x')
            device: Training device ('auto', 'cpu', 'cuda')
        """
        self.dataset_path = Path(dataset_path)
        self.model_size = model_size
        self.device = device
        
        # Training configuration
        self.training_config = {
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640,
            'learning_rate': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box_loss_gain': 0.05,
            'cls_loss_gain': 0.5,
            'obj_loss_gain': 1.0,
            'iou_threshold': 0.20,
            'anchor_threshold': 4.0,
            'fl_gamma': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }
        
        # Setup directories
        self.setup_training_environment()
        
        print(f"🚀 YOLOv5 Trainer initialized")
        print(f"📁 Dataset: {self.dataset_path}")
        print(f"🎯 Model: YOLOv5{model_size}")
        print(f"💻 Device: {device}")
    
    def setup_training_environment(self):
        """Setup training environment and directories"""
        # Create training directories
        self.runs_dir = Path("runs/train")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"custom_yolov5{self.model_size}_{timestamp}"
        self.experiment_dir = self.runs_dir / self.experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        print(f"📂 Experiment directory: {self.experiment_dir}")
    
    def validate_dataset(self):
        """Validate dataset structure and configuration"""
        print("🔍 Validating dataset...")
        
        # Check dataset.yaml
        dataset_yaml = self.dataset_path / "dataset.yaml"
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"dataset.yaml not found in {self.dataset_path}")
        
        # Load dataset configuration
        with open(dataset_yaml, 'r') as f:
            self.dataset_config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ['train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in self.dataset_config:
                raise ValueError(f"Missing required field '{field}' in dataset.yaml")
        
        # Check directories exist
        for split in ['train', 'val']:
            if split in self.dataset_config:
                split_path = self.dataset_path / self.dataset_config[split]
                if not split_path.exists():
                    raise FileNotFoundError(f"Split directory not found: {split_path}")
        
        # Count images and labels
        validation_stats = {}
        for split in ['train', 'val', 'test']:
            if split in self.dataset_config:
                images_dir = self.dataset_path / self.dataset_config[split]
                labels_dir = self.dataset_path / "labels" / split
                
                if images_dir.exists():
                    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
                    label_files = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
                    
                    validation_stats[split] = {
                        'images': len(image_files),
                        'labels': len(label_files)
                    }
        
        print("✅ Dataset validation complete:")
        print(f"  📊 Classes: {self.dataset_config['nc']}")
        print(f"  🏷️  Class names: {self.dataset_config['names']}")
        
        for split, stats in validation_stats.items():
            print(f"  📂 {split.capitalize()}: {stats['images']} images, {stats['labels']} labels")
        
        return validation_stats
    
    def create_training_config(self, custom_config=None):
        """Create training configuration file"""
        if custom_config:
            self.training_config.update(custom_config)
        
        # Create hyperparameter file
        hyp_config = {
            'lr0': self.training_config['learning_rate'],
            'lrf': 0.01,
            'momentum': self.training_config['momentum'],
            'weight_decay': self.training_config['weight_decay'],
            'warmup_epochs': self.training_config['warmup_epochs'],
            'warmup_momentum': self.training_config['warmup_momentum'],
            'warmup_bias_lr': self.training_config['warmup_bias_lr'],
            'box': self.training_config['box_loss_gain'],
            'cls': self.training_config['cls_loss_gain'],
            'obj': self.training_config['obj_loss_gain'],
            'iou_t': self.training_config['iou_threshold'],
            'anchor_t': self.training_config['anchor_threshold'],
            'fl_gamma': self.training_config['fl_gamma'],
            'hsv_h': self.training_config['hsv_h'],
            'hsv_s': self.training_config['hsv_s'],
            'hsv_v': self.training_config['hsv_v'],
            'degrees': self.training_config['degrees'],
            'translate': self.training_config['translate'],
            'scale': self.training_config['scale'],
            'shear': self.training_config['shear'],
            'perspective': self.training_config['perspective'],
            'flipud': self.training_config['flipud'],
            'fliplr': self.training_config['fliplr'],
            'mosaic': self.training_config['mosaic'],
            'mixup': self.training_config['mixup'],
            'copy_paste': self.training_config['copy_paste']
        }
        
        # Save hyperparameter file
        hyp_file = self.experiment_dir / "hyp.yaml"
        with open(hyp_file, 'w') as f:
            yaml.dump(hyp_config, f, default_flow_style=False)
        
        print(f"📄 Training configuration saved: {hyp_file}")
        return hyp_file
    
    def download_yolov5(self):
        """Download YOLOv5 repository if not exists"""
        yolov5_dir = Path("yolov5")
        
        if not yolov5_dir.exists():
            print("📥 Downloading YOLOv5 repository...")
            try:
                subprocess.run([
                    "git", "clone", "https://github.com/ultralytics/yolov5.git"
                ], check=True)
                print("✅ YOLOv5 repository downloaded")
            except subprocess.CalledProcessError:
                print("❌ Failed to download YOLOv5. Please download manually.")
                return False
        
        # Install requirements
        requirements_file = yolov5_dir / "requirements.txt"
        if requirements_file.exists():
            print("📦 Installing YOLOv5 requirements...")
            try:
                subprocess.run([
                    "pip", "install", "-r", str(requirements_file)
                ], check=True)
                print("✅ Requirements installed")
            except subprocess.CalledProcessError:
                print("⚠️  Warning: Could not install all requirements")
        
        return True
    
    def start_training(self, resume=False, pretrained=True):
        """Start YOLOv5 training"""
        print(f"🚀 Starting YOLOv5{self.model_size} training...")
        print("=" * 60)
        
        # Validate dataset
        self.validate_dataset()
        
        # Create training configuration
        hyp_file = self.create_training_config()
        
        # Download YOLOv5 if needed
        if not self.download_yolov5():
            return False
        
        # Prepare training command
        yolov5_dir = Path("yolov5")
        train_script = yolov5_dir / "train.py"
        
        if not train_script.exists():
            print(f"❌ Training script not found: {train_script}")
            return False
        
        # Build training command
        cmd = [
            "python", str(train_script),
            "--data", str(self.dataset_path / "dataset.yaml"),
            "--cfg", f"yolov5{self.model_size}.yaml",
            "--weights", f"yolov5{self.model_size}.pt" if pretrained else "",
            "--name", self.experiment_name,
            "--epochs", str(self.training_config['epochs']),
            "--batch-size", str(self.training_config['batch_size']),
            "--img", str(self.training_config['img_size']),
            "--hyp", str(hyp_file),
            "--device", str(self.device) if self.device != 'auto' else "",
            "--project", str(self.runs_dir.parent),
            "--exist-ok"
        ]
        
        # Remove empty arguments
        cmd = [arg for arg in cmd if arg]
        
        if resume:
            cmd.extend(["--resume", "True"])
        
        print(f"🔧 Training command:")
        print(" ".join(cmd))
        print("=" * 60)
        
        # Start training
        try:
            # Change to yolov5 directory for training
            original_dir = os.getcwd()
            os.chdir(yolov5_dir)
            
            # Run training
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            # Return to original directory
            os.chdir(original_dir)
            
            if result.returncode == 0:
                print("✅ Training completed successfully!")
                self.post_training_analysis()
                return True
            else:
                print("❌ Training failed!")
                return False
                
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False
    
    def post_training_analysis(self):
        """Analyze training results"""
        print("\n📊 Post-training analysis...")
        
        # Find results directory
        results_dir = Path("runs/train") / self.experiment_name
        
        if not results_dir.exists():
            print("❌ Results directory not found")
            return
        
        # Load training results
        results_file = results_dir / "results.csv"
        if results_file.exists():
            self.plot_training_results(results_file)
        
        # Find best model
        best_model = results_dir / "weights" / "best.pt"
        last_model = results_dir / "weights" / "last.pt"
        
        if best_model.exists():
            print(f"🏆 Best model saved: {best_model}")
        if last_model.exists():
            print(f"💾 Last model saved: {last_model}")
        
        # Copy models to experiment directory
        if best_model.exists():
            shutil.copy2(best_model, self.experiment_dir / "best.pt")
        if last_model.exists():
            shutil.copy2(last_model, self.experiment_dir / "last.pt")
        
        print(f"📁 Training artifacts saved to: {self.experiment_dir}")
    
    def plot_training_results(self, results_file):
        """Plot training metrics"""
        try:
            import pandas as pd
            
            # Load results
            df = pd.read_csv(results_file)
            df.columns = df.columns.str.strip()  # Remove whitespace
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'YOLOv5{self.model_size} Training Results', fontsize=16)
            
            # Loss plots
            if 'train/box_loss' in df.columns:
                axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss')
                axes[0, 0].plot(df['epoch'], df['train/obj_loss'], label='Obj Loss')
                axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Cls Loss')
                axes[0, 0].set_title('Training Losses')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # mAP plots
            if 'metrics/mAP_0.5' in df.columns:
                axes[0, 1].plot(df['epoch'], df['metrics/mAP_0.5'], label='mAP@0.5')
                axes[0, 1].plot(df['epoch'], df['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95')
                axes[0, 1].set_title('Mean Average Precision')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('mAP')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Precision/Recall
            if 'metrics/precision' in df.columns:
                axes[1, 0].plot(df['epoch'], df['metrics/precision'], label='Precision')
                axes[1, 0].plot(df['epoch'], df['metrics/recall'], label='Recall')
                axes[1, 0].set_title('Precision & Recall')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Learning rate
            if 'lr/pg0' in df.columns:
                axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='LR pg0')
                axes[1, 1].plot(df['epoch'], df['lr/pg1'], label='LR pg1')
                axes[1, 1].plot(df['epoch'], df['lr/pg2'], label='LR pg2')
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.experiment_dir / "training_results.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"📈 Training plots saved: {plot_file}")
            
        except Exception as e:
            print(f"⚠️  Could not plot results: {e}")
    
    def optimize_hyperparameters(self, trials=10):
        """Hyperparameter optimization using Optuna"""
        print(f"🔧 Starting hyperparameter optimization ({trials} trials)...")
        
        try:
            import optuna
            
            def objective(trial):
                # Suggest hyperparameters
                lr = trial.suggest_float('lr0', 0.001, 0.1, log=True)
                momentum = trial.suggest_float('momentum', 0.8, 0.99)
                weight_decay = trial.suggest_float('weight_decay', 0.0001, 0.001, log=True)
                
                # Update training config
                custom_config = {
                    'learning_rate': lr,
                    'momentum': momentum,
                    'weight_decay': weight_decay,
                    'epochs': 50  # Reduced for optimization
                }
                
                # Create temporary trainer
                temp_trainer = YOLOv5Trainer(self.dataset_path, self.model_size, self.device)
                temp_trainer.training_config.update(custom_config)
                
                # Run training
                success = temp_trainer.start_training(pretrained=True)
                
                if success:
                    # Return mAP as objective (to maximize)
                    results_dir = Path("runs/train") / temp_trainer.experiment_name
                    results_file = results_dir / "results.csv"
                    
                    if results_file.exists():
                        import pandas as pd
                        df = pd.read_csv(results_file)
                        if 'metrics/mAP_0.5' in df.columns:
                            return df['metrics/mAP_0.5'].max()
                
                return 0.0  # Return 0 if training failed
            
            # Create study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=trials)
            
            # Print best parameters
            print("🏆 Best hyperparameters:")
            for key, value in study.best_params.items():
                print(f"  {key}: {value}")
            
            print(f"🎯 Best mAP: {study.best_value:.4f}")
            
            # Save optimization results
            optimization_file = self.experiment_dir / "hyperparameter_optimization.json"
            with open(optimization_file, 'w') as f:
                json.dump({
                    'best_params': study.best_params,
                    'best_value': study.best_value,
                    'n_trials': len(study.trials)
                }, f, indent=2)
            
            return study.best_params
            
        except ImportError:
            print("❌ Optuna not installed. Install with: pip install optuna")
            return None
        except Exception as e:
            print(f"❌ Hyperparameter optimization failed: {e}")
            return None
    
    def evaluate_model(self, model_path=None, test_data=None):
        """Evaluate trained model"""
        if model_path is None:
            model_path = self.experiment_dir / "best.pt"
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return
        
        print(f"📊 Evaluating model: {model_path}")
        
        # Load model for evaluation
        try:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path))
            
            # Test on sample images
            if test_data:
                results = model(test_data)
                results.show()
                results.save()
            
            print("✅ Model evaluation complete")
            
        except Exception as e:
            print(f"❌ Model evaluation failed: {e}")

def main():
    """Demonstrate custom training pipeline"""
    print("🚀 YOLOv5 Custom Training Pipeline")
    print("=" * 60)
    
    # Example usage
    dataset_path = "datasets/custom_objects"
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("💡 Please run dataset_preparation.py first to create a dataset")
        return
    
    # Initialize trainer
    trainer = YOLOv5Trainer(
        dataset_path=dataset_path,
        model_size='s',  # Start with small model for faster training
        device='auto'
    )
    
    # Custom training configuration
    custom_config = {
        'epochs': 50,  # Reduced for demo
        'batch_size': 8,  # Smaller batch size for demo
        'img_size': 416,  # Smaller image size for faster training
        'learning_rate': 0.01
    }
    
    print("\n🔧 Training Configuration:")
    for key, value in custom_config.items():
        print(f"  {key}: {value}")
    
    # Start training
    print(f"\n🚀 Starting training...")
    success = trainer.start_training(pretrained=True)
    
    if success:
        print("\n✅ Training completed successfully!")
        print(f"📁 Results saved to: {trainer.experiment_dir}")
        
        # Optional: Run hyperparameter optimization
        optimize = input("\n🔧 Run hyperparameter optimization? (y/n): ").lower() == 'y'
        if optimize:
            best_params = trainer.optimize_hyperparameters(trials=5)
            if best_params:
                print("🏆 Optimization complete!")
    else:
        print("❌ Training failed!")
    
    print("\n💡 Next steps:")
    print("1. Review training results and plots")
    print("2. Test the trained model on new images")
    print("3. Fine-tune hyperparameters if needed")
    print("4. Deploy the model for real-time detection")

if __name__ == "__main__":
    main()
