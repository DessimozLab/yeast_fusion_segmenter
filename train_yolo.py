#!/usr/bin/env python3

import os
import yaml
import argparse
from ultralytics import YOLO
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('YOLOTrainer')

def parse_args():
    """Parse command line arguments for YOLO model training"""
    parser = argparse.ArgumentParser(description='Train YOLO model on custom dataset')
    
    # Dataset arguments
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument('--data', type=str,
                               help='Path to dataset YAML configuration file')
    dataset_group.add_argument('--dataset', type=str,
                               help='Dataset-info JSON saved by MicroscopyImageDataset')
    parser.add_argument('--annotated-only', action='store_true',
                        help='Use only images with non-empty annotation labels')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate the pretrained/model checkpoint instead of training')
    parser.add_argument('--eval-split', choices=['train', 'val', 'test'], default='test',
                        help='Dataset split to evaluate with --evaluate (default: test)')
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
    parser.add_argument('--notebook-protocol', action='store_true',
                        help='Use the final segment_retrain(1).ipynb YOLOv8s training protocol')
    parser.add_argument('--zoom-augmentation', action='store_true',
                        help='Use stronger random scale/translation so training views include zoomed crops')
    
    return parser.parse_args()

def notebook_hyperparameters():
    """Return the final YOLO hyperparameters used by the retraining notebook."""
    return {
        'lr0': 0.001,
        'lrf': 0.0001,
        'momentum': 0.5,
        'weight_decay': 0.0001,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.01,
        'box': 10,
        'cls': 5,
        'dfl': 0.5,
        'pose': 0,
        'kobj': 0,
        'label_smoothing': 0.0,
        'nbs': 32,
        'hsv_h': 0.01,
        'hsv_s': 0.01,
        'hsv_v': 0.01,
        'degrees': 180.0,
        'translate': 0.1,
        'scale': 0.1,
        'shear': 0.1,
        'perspective': 0.0,
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 0.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
    }


def load_hyperparameters(hyp_file=None, notebook_protocol=False):
    """Load hyperparameters from file or use defaults"""
    if hyp_file and os.path.exists(hyp_file):
        logger.info(f"Loading hyperparameters from {hyp_file}")
        with open(hyp_file, 'r') as f:
            return yaml.safe_load(f)
    
    if notebook_protocol:
        logger.info("Using final segment_retrain(1).ipynb hyperparameters")
        return notebook_hyperparameters()

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


def annotated_only_yaml(data_yaml: str) -> str:
    """Create a YOLO YAML whose splits contain only non-empty label files.

    Dataset preparation pairs each generated label with the HDF5 path stored in
    the dataset-info record.  A non-empty YOLO label is therefore the usable
    model-facing representation of an associated annotation.
    """
    data_path = Path(data_yaml).resolve()
    config = yaml.safe_load(data_path.read_text())
    root = Path(config.get('path', data_path.parent)).resolve()
    filtered = root / 'annotated_only'
    filtered.mkdir(parents=True, exist_ok=True)

    for split in ('train', 'val', 'test'):
        image_dir, label_dir = root / split / 'images', root / split / 'labels'
        images = []
        for image in sorted(image_dir.glob('*.png')):
            label = label_dir / f'{image.stem}.txt'
            if label.exists() and label.read_text().strip():
                images.append(str(image.resolve()))
        list_path = filtered / f'{split}.txt'
        list_path.write_text('\n'.join(images) + ('\n' if images else ''))
        config[split] = str(list_path.resolve())

    output = filtered / 'dataset.yaml'
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    return str(output)


def resolve_data_path(args) -> str:
    """Resolve dataset-info/YOLO YAML input and apply optional label filtering."""
    data_path = args.data
    if args.dataset:
        from image_dataset import MicroscopyImageDataset
        dataset_info = MicroscopyImageDataset.load_dataset_info(args.dataset)
        data_path = dataset_info['yolo_data']
        if not data_path:
            raise ValueError(f"Dataset info has no prepared YOLO data path: {args.dataset}")
        logger.info("Using dataset '%s' from %s", dataset_info['name'], args.dataset)
    if args.annotated_only:
        data_path = annotated_only_yaml(data_path)
        logger.info("Using annotated-only dataset YAML: %s", data_path)
    return data_path

def train_model(args):
    """Main function to train the YOLO model"""
    # Validate model path
    if not os.path.exists(args.model) and not args.model.startswith('yolov8'):
        raise FileNotFoundError(f"Model not found: {args.model}")
    
    # Resolve a persisted dataset definition when requested.
    data_path = resolve_data_path(args)

    # Validate and load dataset configuration
    data_config = validate_dataset(data_path)
    
    # Load hyperparameters
    hyp = load_hyperparameters(args.hyp, notebook_protocol=args.notebook_protocol)
    if args.zoom_augmentation:
        # Ultralytics RandomPerspective applies a scale about the image centre;
        # portions outside the 1024px canvas are cropped.  This is an online
        # zoom/crop augmentation that preserves image/mask geometry jointly.
        hyp = dict(hyp)
        hyp.update({'scale': 0.5, 'translate': 0.2})
        logger.info("Using zoom/crop augmentation: scale=0.5, translate=0.2")
    if args.notebook_protocol:
        # These are the final training settings in the notebook. Explicit CLI
        # values remain honored so a user can run a shorter smoke test.
        if args.model == 'yolov8n-seg.pt':
            args.model = 'yolov8s-seg.pt'
        if args.epochs == 100:
            args.epochs = 1000
        if args.batch_size == 8:
            args.batch_size = 20
        if args.workers == 4:
            args.workers = 8
    
    # Initialize the model
    logger.info(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Training settings
    logger.info(f"Starting training for {args.epochs} epochs")
    results = model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        project='yolo_training',
        name=os.path.splitext(args.output)[0],
        # Never reuse a prior run directory: an interrupted run can leave a
        # partial results.csv and make Ultralytics' final plotting fail.
        exist_ok=False,
        pretrained=True,
        **hyp
    )
    
    # Copy the best training checkpoint to the requested handoff location.
    output_path = args.output
    logger.info(f"Saving model to {output_path}")
    best_path = os.path.join(str(model.trainer.save_dir), 'weights', 'best.pt')
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Training completed but best checkpoint was not found: {best_path}")
    if os.path.abspath(best_path) != os.path.abspath(output_path):
        import shutil
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        shutil.copy2(best_path, output_path)
    
    return results


def evaluate_model(args):
    """Evaluate a pretrained or trained model without modifying its weights."""
    if not os.path.exists(args.model) and not args.model.startswith('yolov8'):
        raise FileNotFoundError(f"Model not found: {args.model}")
    data_path = resolve_data_path(args)
    validate_dataset(data_path)
    model = YOLO(args.model)
    logger.info("Evaluating %s on %s (%s split)", args.model, data_path, args.eval_split)
    return model.val(
        data=data_path, split=args.eval_split, imgsz=args.img_size,
        batch=args.batch_size, device=args.device, workers=args.workers,
        project='yolo_evaluation', name=Path(args.model).stem, exist_ok=False,
    )

if __name__ == "__main__":
    args = parse_args()
    if args.evaluate:
        evaluate_model(args)
        logger.info("Evaluation completed successfully")
    else:
        train_model(args)
        logger.info("Training completed successfully")
        print(f"Trained model saved to: {args.output}")
