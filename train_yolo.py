#!/usr/bin/env python3

import os
import yaml
import argparse
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('YOLOTrainer')

def parse_args():
    """Parse command line arguments for YOLO model training"""
    parser = argparse.ArgumentParser(description='Train YOLO model on custom dataset')
    
    # Dataset arguments
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to dataset YAML configuration file')
    parser.add_argument('--img-size', type=int, default=1024, 
                        help='Input image size (default: 1024)')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='yolov8n-seg.pt', 
                        help='Path to base model for fine-tuning (default: yolov8n-seg.pt)')
    parser.add_argument('--output', type=str, default='yolov8_retrained.pt', 
                        help='Name for the output model file (default: yolov8_retrained.pt)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=8, 
                        help='Training batch size (default: 8)')
    parser.add_argument('--device', type=str, default='0', 
                        help='Device to run training on (default: 0 for first GPU)')
    parser.add_argument('--workers', type=int, default=4, 
                        help='Number of worker threads (default: 4)')
    parser.add_argument('--hyp', type=str, default=None, 
                        help='Path to hyperparameter file (default: None)')
    
    return parser.parse_args()

def load_hyperparameters(hyp_file=None):
    """Load hyperparameters from file or use defaults"""
    if hyp_file and os.path.exists(hyp_file):
        logger.info(f"Loading hyperparameters from {hyp_file}")
        with open(hyp_file, 'r') as f:
            return yaml.safe_load(f)
    
    # Default hyperparameters
    logger.info("Using default hyperparameters")
    return {
        'lr0': 0.001,             # initial learning rate
        'lrf': 0.0001,            # final learning rate (lr0 * lrf)
        'momentum': 0.5,          # SGD momentum/Adam beta1
        'weight_decay': 0.0001,   # optimizer weight decay
        'warmup_epochs': 3.0,     # warmup epochs
        'warmup_momentum': 0.8,   # warmup initial momentum
        'warmup_bias_lr': 0.01,   # warmup initial bias lr
        'box': 10,                # box loss gain
        'cls': 5,                 # cls loss gain
        'dfl': 0.5,               # dfl loss gain
        'label_smoothing': 0.0,   # label smoothing
        'nbs': 64,                # nominal batch size
        'hsv_h': 0.01,            # HSV-Hue augmentation
        'hsv_s': 0.01,            # HSV-Saturation augmentation
        'hsv_v': 0.01,            # HSV-Value augmentation
        'degrees': 180.0,         # rotation (+/- deg)
        'translate': 0.1,         # translation (+/- fraction)
        'scale': 0.1,             # scale (+/- gain)
        'shear': 0.1,             # shear (+/- deg)
        'perspective': 0.0,       # perspective (+/- fraction)
        'flipud': 0.5,            # flip up-down (probability)
        'fliplr': 0.5,            # flip left-right (probability)
        'mosaic': 0.2,            # mosaic (probability)
        'mixup': 0.0,             # mixup (probability)
    }

def validate_dataset(data_yaml):
    """Validate that the dataset is correctly formatted"""
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset configuration file not found: {data_yaml}")
    
    # Read YAML file
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Check required paths
    required_paths = ['train', 'val']
    for path in required_paths:
        if path not in data_config:
            raise ValueError(f"Dataset YAML missing required path: {path}")
        
        # Check if path exists (handle relative paths)
        dataset_dir = os.path.dirname(os.path.abspath(data_yaml))
        full_path = os.path.join(dataset_dir, data_config[path])
        
        if not os.path.exists(full_path):
            logger.warning(f"Warning: {path} path does not exist: {full_path}")
    
    # Check class names
    if 'names' not in data_config:
        raise ValueError("Dataset YAML missing 'names' field for class names")
    
    logger.info(f"Dataset validated with {len(data_config['names'])} classes")
    return data_config

def train_model(args):
    """Main function to train the YOLO model"""
    # Validate model path
    if not os.path.exists(args.model) and not args.model.startswith('yolov8'):
        raise FileNotFoundError(f"Model not found: {args.model}")
    
    # Validate and load dataset configuration
    data_config = validate_dataset(args.data)
    
    # Load hyperparameters
    hyp = load_hyperparameters(args.hyp)
    
    # Initialize the model
    logger.info(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Training settings
    logger.info(f"Starting training for {args.epochs} epochs")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        project='yolo_training',
        name=os.path.splitext(args.output)[0],
        exist_ok=True,
        pretrained=True,
        **hyp
    )
    
    # Save the model
    output_path = args.output
    logger.info(f"Saving model to {output_path}")
    model.export(format='pt')
    
    # Move the exported model to the target path if it's not already there
    export_path = f"{model.trainer.save_dir}/{model.trainer.name}/weights/best.pt"
    if os.path.exists(export_path) and export_path != output_path:
        import shutil
        shutil.copy(export_path, output_path)
    
    return results

if __name__ == "__main__":
    args = parse_args()
    train_model(args)
    logger.info("Training completed successfully")
    print(f"Trained model saved to: {args.output}")
