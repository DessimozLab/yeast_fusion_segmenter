"""
Yeast Fusion Segmenter

A deep learning-based tool for segmenting yeast cells in fusion experiments using YOLOv8.

This package provides tools for:
- Training YOLOv8 models on yeast cell datasets
- Batch prediction on microscopy images
- Data preprocessing and augmentation
- Statistical analysis of segmented cells

Main modules:
- train_yolo: Training pipeline for YOLOv8 models
- batch_predict: Batch inference on image datasets  
- prepare_yolo_data: Data preparation utilities

Example usage:
    from ultralytics import YOLO
    model = YOLO('yolov8n-seg_yfusion.pt')
    results = model.predict('image.png')
"""

__version__ = "0.1.0"
__author__ = "DessimozLab"
__email__ = "your.email@example.com"

# Import main functions for convenience
try:
    from .train_yolo import train_model, load_hyperparameters
    from .batch_predict import predict_and_collect, process_image
    from .annotate_images import main as annotate_main
except ImportError:
    # Handle case where dependencies aren't installed
    pass

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "train_model",
    "load_hyperparameters", 
    "predict_and_collect",
    "process_image",
    "annotate_main",
]