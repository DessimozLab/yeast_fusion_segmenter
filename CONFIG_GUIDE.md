# Configuration Files Guide

This guide explains how to use YAML configuration files with the prediction scripts.

## Overview

Both `batch_predict.py` and `annotate_images.py` support YAML configuration files, making it easier to:
- Store and reuse common parameter sets
- Share configurations across team members
- Version control your prediction settings
- Reduce command-line complexity

## Configuration Files

### batch_predict_config.yaml

Used with `batch_predict.py` for batch processing of images.

**Key Parameters:**
- `model`: Path to YOLO model (.pt file)
- `input_dir`: Directory containing input images
- `output_csv`: Output CSV file path
- `format`: Image format ('png', 'tif', or 'czi')
- `crop`: Crop size for processing
- `zoom`: Enable overlapping crop predictions
- `zoom_factor`: Zoom factor for cropping (0.0-1.0)

### annotate_images_config.yaml

Used with `annotate_images.py` for annotating images with segmentation results.

**Key Parameters:**
- `model`: Path to YOLO model (.pt file)
- `input`: Input directory containing images
- `output`: Output CSV file path
- `format`: Image format ('auto', 'tiff', 'czi', 'single')
- `confidence`: Confidence threshold (0.0-1.0)
- `imgsz`: Image size for inference
- `crop`: Crop size for input images
- `zoom`: Enable overlapping crop predictions
- `zoom_factor`: Zoom factor for cropping
- `verbose`: Enable verbose output

## Usage

### Basic Usage

Run with config file only:
```bash
python batch_predict.py --config batch_predict_config.yaml
python annotate_images.py --config annotate_images_config.yaml
```

### Override Parameters

Config values can be overridden from the command line:
```bash
# Override crop size
python batch_predict.py --config batch_predict_config.yaml --crop 512

# Override confidence threshold and enable zoom
python annotate_images.py --config annotate_images_config.yaml --confidence 0.7 --zoom

# Override multiple parameters
python batch_predict.py --config batch_predict_config.yaml \
  --input_dir new_data/ \
  --output_csv new_results.csv \
  --zoom
```

### Hybrid Approach

Mix config file and command-line arguments:
```bash
# Use config for most settings, specify input/output on command line
python batch_predict.py --config batch_predict_config.yaml \
  --input_dir /path/to/new/data \
  --output_csv /path/to/results.csv
```

## Creating Custom Configurations

### Example: High-Resolution Image Processing

Create a config for high-res images with zoom mode:

**high_res_config.yaml:**
```yaml
model: yolov8s-seg_yfusion.pt
input_dir: high_res_images/
output_csv: high_res_results.csv
format: tif
crop: 1024
zoom: true
zoom_factor: 0.5  # Smaller = more crops
```

Usage:
```bash
python batch_predict.py --config high_res_config.yaml
```

### Example: Low-Confidence Exploration

Create a config for finding all potential cells:

**low_confidence_config.yaml:**
```yaml
model: yolov8s-seg_yfusion.pt
input: exploratory_images/
output: all_detections.csv
format: auto
confidence: 0.3  # Lower threshold
imgsz: 1024
verbose: true  # See what's being detected
```

Usage:
```bash
python annotate_images.py --config low_confidence_config.yaml
```

### Example: Production Pipeline

Create a config for production use:

**production_config.yaml:**
```yaml
model: yolov8s-seg_yfusion.pt
input: /data/microscopy/batch_001/
output: /results/batch_001_results.csv
format: auto
confidence: 0.8  # High confidence only
imgsz: 1024
crop: 1024
zoom: false
verbose: false
```

Usage:
```bash
python annotate_images.py --config production_config.yaml
```

## Best Practices

### 1. Version Control Configs
Store config files in git to track parameter changes:
```bash
git add batch_predict_config.yaml
git commit -m "Update model to yolov8s-seg_yfusion.pt"
```

### 2. Name Configs Descriptively
Use clear names that indicate the purpose:
- `high_res_zoom_config.yaml`
- `fast_screening_config.yaml`
- `production_final_config.yaml`

### 3. Document Custom Configs
Add comments explaining non-standard settings:
```yaml
model: yolov8s-seg_yfusion.pt
input_dir: test_images/
output_csv: results.csv
format: png
crop: 512  # Smaller crop for faster testing
zoom: true
zoom_factor: 0.8  # Less overlap for speed
```

### 4. Validate Before Production
Test configs on small datasets first:
```bash
# Test on a few images
python batch_predict.py --config production_config.yaml \
  --input_dir test_subset/
```

### 5. Keep Defaults Explicit
Specify all important parameters even if they match defaults:
```yaml
# Good - explicit
confidence: 0.5
crop: 1024
zoom: false

# Avoid - relying on implicit defaults
# (harder to understand when revisiting)
```

## Troubleshooting

### Config file not found
```
Error: FileNotFoundError: batch_predict_config.yaml
```
Solution: Use absolute path or ensure you're in the correct directory:
```bash
python batch_predict.py --config /full/path/to/config.yaml
```

### Missing required parameters
```
Error: Missing required arguments: input_dir, model
```
Solution: Ensure config file contains all required fields:
```yaml
# Required for batch_predict.py
model: yolov8s-seg_yfusion.pt
input_dir: images/
output_csv: results.csv
format: png
```

### Invalid parameter values
```
Error: argument --confidence: invalid choice: 1.5
```
Solution: Check value ranges in config:
- `confidence`: 0.0-1.0
- `zoom_factor`: 0.0-1.0
- `crop`: positive integer
- `format`: 'png', 'tif', 'czi' (batch_predict) or 'auto', 'tiff', 'czi', 'single' (annotate_images)

### YAML syntax errors
```
Error: yaml.scanner.ScannerError
```
Solution: Validate YAML syntax:
```bash
# Install yamllint
pip install yamllint

# Check config file
yamllint batch_predict_config.yaml
```

## Advanced Usage

### Environment-Specific Configs

Create different configs for different environments:

**dev_config.yaml:**
```yaml
model: yolov8n-seg_yfusion.pt  # Smaller/faster model for development
input: test_data/
output: dev_results.csv
verbose: true
```

**prod_config.yaml:**
```yaml
model: yolov8s-seg_yfusion.pt  # Production model
input: /data/production/
output: /results/production_results.csv
verbose: false
```

### Batch Processing Multiple Configs

Process different datasets with different configs:
```bash
#!/bin/bash
# process_all.sh

python batch_predict.py --config dataset1_config.yaml
python batch_predict.py --config dataset2_config.yaml
python batch_predict.py --config dataset3_config.yaml
```

### Config Templates

Create template configs for common use cases:
```bash
# templates/
├── template_high_res.yaml
├── template_fast_screening.yaml
└── template_production.yaml

# Copy and customize
cp templates/template_high_res.yaml my_experiment_config.yaml
# Edit my_experiment_config.yaml with specific paths
```

## Migration from Command-Line Only

Convert existing command-line workflows to configs:

**Before:**
```bash
python batch_predict.py \
  --input_dir datasets/test/images \
  --model yolov8s-seg_yfusion.pt \
  --format png \
  --output_csv results.csv \
  --crop 1024 \
  --zoom \
  --zoom_factor 0.667
```

**After:**
Create `my_config.yaml`:
```yaml
model: yolov8s-seg_yfusion.pt
input_dir: datasets/test/images
output_csv: results.csv
format: png
crop: 1024
zoom: true
zoom_factor: 0.667
```

Then run:
```bash
python batch_predict.py --config my_config.yaml
```

## See Also

- [tests/README.md](tests/README.md) - Testing configuration files
- [README.MD](README.MD) - Main project documentation
- [batch_predict.py](batch_predict.py) - Batch prediction script
- [annotate_images.py](annotate_images.py) - Annotation script
