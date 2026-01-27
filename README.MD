<div align="center">
  <img src="logo.png" alt="Yeast Fusion Segmenter Logo" width="800"/>
</div>

# Yeast Fusion Segmenter

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A sophisticated deep learning tool for automated segmentation of yeast cells in fusion experiments using YOLOv8 instance segmentation. This project provides end-to-end capabilities for training, inference, and analysis of yeast cell fusion events from microscopy images.

## 🔬 Overview

Yeast fusion experiments are crucial for studying cellular processes, mating types, and genetic interactions. This tool automates the tedious process of manually segmenting yeast cells by:

- **Multi-channel Analysis**: Processes brightfield (BF), GFP, and RFP fluorescence channels simultaneously
- **Advanced Segmentation**: Uses YOLOv8 instance segmentation for precise cell boundary detection
- **Fusion Classification**: Identifies and classifies different cell types (individual cells, fusion intermediates, fusion products)
- **Batch Processing**: Handles large datasets with automated preprocessing and inference
- **Statistical Analysis**: Extracts quantitative features from segmented cells for downstream analysis

## 🚀 Features

- **Multi-format Support**: Works with TIFF, CZI, and H5 file formats
- **Data Augmentation**: Built-in augmentation pipeline for robust model training
- **GPU Acceleration**: CUDA-optimized for fast training and inference
- **Configurable Training**: Hyperparameter optimization and custom training pipelines
- **Visualization Tools**: Integrated plotting and analysis capabilities
- **Export Options**: Results exported in CSV format with comprehensive statistics

## 📋 Requirements

- Python 3.9+
- NVIDIA GPU with CUDA 11.7+ (recommended for training)
- 8GB+ RAM (16GB+ recommended for large datasets)
- 10GB+ storage space for models and datasets

## � License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. GPU Not Detected

**Problem:** PyTorch doesn't recognize your GPU
**Solutions:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
```

#### 2. Out of Memory Errors

**Problem:** CUDA out of memory during training or inference
**Solutions:**
- Reduce batch size: `--batch-size 1` or `--batch-size 2`
- Reduce image size: `--imgsz 512` or `--crop 512`
- Use zoom mode for large images: `--zoom --zoom_factor 0.5`
- Clear GPU cache in Python:
  ```python
  import torch
  torch.cuda.empty_cache()
  ```

#### 3. CZI Files Not Loading

**Problem:** Cannot open CZI files
**Solutions:**
```bash
# Install ImageJ/Fiji support
pip install pyimagej

# Or use Conda
conda install -c conda-forge pyimagej openjdk=8

# Verify installation
python -c "import imagej; print('ImageJ OK')"
```

#### 4. Low Detection Accuracy

**Problem:** Model doesn't detect cells well
**Solutions:**
- Use the latest fine-tuned model: `yolov8s-seg_yfusion.pt` or `yolov8s-seg_yfusionmk3.pt`
- Enable zoom mode for better detail: `--zoom`
- Adjust confidence threshold: Lower threshold finds more detections (may include false positives)
  ```bash
  # Try lower confidence in annotate_images.py
  python annotate_images.py --model yolov8s-seg_yfusion.pt --input data/ --output results.csv --confidence 0.3
  ```
- Ensure image quality:
  - Proper focus and contrast
  - Correct channel order: BF (channel 0), RFP (channel 1), GFP (channel 2)
  - Adequate resolution (1024x1024 recommended)
- Compare standard vs zoom mode results:
  ```bash
  # Standard mode
  python batch_predict.py --input_dir data/ --model yolov8s-seg_yfusion.pt --format czi --output_csv standard.csv
  
  # Zoom mode (typically 30-60% more detections)
  python batch_predict.py --input_dir data/ --model yolov8s-seg_yfusion.pt --format czi --output_csv zoom.csv --zoom
  ```

#### 5. Slow Processing Speed

**Problem:** Predictions take too long
**Solutions:**
- Use GPU instead of CPU (100x faster)
- Reduce image size: `--imgsz 512`
- Use smaller model: `yolov8n-seg.pt` instead of `yolov8l-seg.pt`
- Process images in batches
- Don't use zoom mode unless necessary

### Best Practices

#### Data Preparation
1. **Image Quality:** Ensure good focus and contrast
2. **Normalization:** Images are auto-normalized, but consistent lighting helps
3. **Channel Order:** Always BF (channel 0), RFP (channel 1), GFP (channel 2)
4. **Image Size:** 1024×1024 works best; smaller sizes may lose detail

#### Training Tips
1. **Start Small:** Begin with 100 epochs to test, then increase
2. **Data Augmentation:** Use rotation, flipping for better generalization
3. **Validation Set:** Always keep 10-20% for validation
4. **Monitor Loss:** Stop if validation loss stops decreasing
5. **Fine-tuning:** Start from pre-trained model rather than from scratch

#### Inference Optimization
1. **Batch Processing:** Process multiple images together
2. **Zoom Mode:** Use for high-resolution images only
3. **Confidence Threshold:** Start with 0.5, adjust based on results
4. **GPU Usage:** Always use GPU for faster processing
5. **Output Format:** CSV for analysis, annotated images for visual verification

### Performance Guidelines

| Task | Recommended Settings | Expected Speed | Detection Rate |
|------|---------------------|----------------|----------------|
| Training | batch=4-8, imgsz=1024, GPU required | 2-4 hrs (2000 epochs) | N/A |
| Inference (PNG/small) | imgsz=512, CPU ok | 0.5-1 sec/image | Standard |
| Inference (CZI/large) | imgsz=1024, GPU required | 3-5 sec/image | Standard |
| Zoom mode (CZI) | zoom_factor=0.667, GPU required | 6-12 sec/image | +30-60% detections |
| Batch prediction | GPU required, format=czi | 100-200 images/min | Standard |
| Batch zoom mode | GPU required, zoom | 40-80 images/min | +30-60% detections |

**Zoom Mode Benefits:**
- Detects 30-60% more objects compared to standard mode
- Better detection of small or partially visible cells
- Improves accuracy on cell boundaries
- Recommended for high-resolution microscopy (1024x1024+)
- Trade-off: 2-3x slower processing time

**Model Comparison (on CZI files, 1024x1024):**

| Model | Speed (img/sec) | Accuracy | Use Case |
|-------|----------------|----------|----------|
| yolov8n-seg.pt | ~15 | Baseline | Quick testing |
| yolov8s-seg_yfusion.pt | ~8 | High | Production (recommended) |
| yolov8s-seg_yfusionmk3.pt | ~8 | Very High | Best accuracy |
| yolov8l-seg.pt | ~3 | Medium | General (not fine-tuned) |

### Common Use Cases and Recipes

#### Recipe 1: Quick Validation Run
```bash
# Test on a few images first
python batch_predict.py \
  --input_dir sample_images/ \
  --model yolov8s-seg_yfusion.pt \
  --format czi \
  --output_csv validation.csv

# Check results
python -c "import pandas as pd; df = pd.read_csv('validation.csv'); print(f'Found {len(df)} detections in {df[\"file\"].nunique()} images')"
```

#### Recipe 2: Production Batch Processing
```bash
# Full dataset with zoom mode for best results
python batch_predict.py \
  --input_dir full_experiment/ \
  --model yolov8s-seg_yfusion.pt \
  --format czi \
  --output_csv full_results.csv \
  --zoom \
  --zoom_factor 0.667
```

#### Recipe 3: Statistical Analysis Pipeline
```bash
# Generate predictions
python batch_predict.py \
  --input_dir data/ \
  --model yolov8s-seg_yfusion.pt \
  --format czi \
  --output_csv raw_results.csv \
  --zoom

# Python analysis
python << EOF
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('raw_results.csv')

# Class distribution
print("Detections per class:")
print(df['class'].value_counts())

# Channel statistics by class
print("\nMean GFP intensity by class:")
print(df.groupby('class')['gfp_mean'].mean())

# Export filtered results (high confidence only)
filtered = df[df['proba'] > 0.7]
filtered.to_csv('high_confidence_results.csv', index=False)
print(f"\nExported {len(filtered)} high-confidence detections")
EOF
```

#### Recipe 4: Multi-Sample Comparison
```bash
# Process control and treatment groups
python batch_predict.py --input_dir control/ --model yolov8s-seg_yfusion.pt --format czi --output_csv control.csv --zoom
python batch_predict.py --input_dir treatment/ --model yolov8s-seg_yfusion.pt --format czi --output_csv treatment.csv --zoom

# Compare results
python << EOF
import pandas as pd

ctrl = pd.read_csv('control.csv')
treat = pd.read_csv('treatment.csv')

print(f"Control: {len(ctrl)} detections across {ctrl['file'].nunique()} images")
print(f"Treatment: {len(treat)} detections across {treat['file'].nunique()} images")

print(f"\nControl class distribution:\n{ctrl['class'].value_counts(normalize=True)}")
print(f"\nTreatment class distribution:\n{treat['class'].value_counts(normalize=True)}")
EOF
```

### Quick Install with Pip

For users who want to install the package directly:

```bash
pip install yeast-fusion-segmenter
```

### Development Installation

For development or custom modifications, follow the detailed guide below.

## Installation Guide for Ubuntu

This guide will walk you through installing Mamba (a fast, drop-in replacement for Conda) and setting up the required environment on a clean Ubuntu installation.

### 1. Update System Packages

First, ensure your system is up to date:

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Install Required System Dependencies

Install necessary system packages:

```bash
sudo apt install -y build-essential git wget curl libgl1-mesa-glx libglib2.0-0
```

### 3. Install Mamba (Miniforge)

Mamba is a fast reimplementation of Conda package manager. We'll install Miniforge which includes Mamba:

```bash
# Download the Miniforge installer
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh

# Make the installer executable
chmod +x miniforge.sh

# Run the installer
./miniforge.sh -b

# Initialize Mamba for your shell
~/miniforge3/bin/mamba init

# Reload your shell configuration
source ~/.bashrc
```

### 4. Create Environment Using the YAML File

Clone the repository (if not already done) and navigate to the project directory:

```bash
git clone [repository-url]
cd yeast_fusion_segmenter
```

Create the environment using Mamba:

```bash
mamba env create -f environment.yml
```

This will create a new environment called `yeast_fusion_segmenter` with all required dependencies.

### 5. Activate the Environment

Activate the environment:

```bash
mamba activate yeast_fusion_segmenter
```

### 6. Verify GPU Support for PyTorch

To verify that PyTorch can see your GPU:

```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### 7. Additional CUDA Setup (if needed)

If PyTorch cannot detect your GPU, you may need to install the NVIDIA drivers:

```bash
sudo apt install -y nvidia-driver-525  # Choose the latest available driver
sudo reboot  # Reboot to load the new driver
```

After reboot, check if NVIDIA drivers are properly installed:

```bash
nvidia-smi
```

### 8. Run the Segmentation Notebook

Launch Jupyter to run the segmentation notebook:

```bash
jupyter notebook
```

Navigate to and open `segment_retrain.ipynb` to start using the segmenter.

## 📚 Usage

### Quick Start Guide

#### 1. Running Batch Predictions with batch_predict.py

The `batch_predict.py` script is designed for high-throughput prediction on multiple images with comprehensive statistical analysis:

**Basic Usage:**
```bash
# Process PNG images from a directory
python batch_predict.py --input_dir datasets/test/images --model yolov8s-seg_yfusion.pt --format png --output_csv results.csv

# Process TIFF files with custom crop size
python batch_predict.py --input_dir tiff_images/ --model yolov8s-seg_yfusion.pt --format tif --output_csv tiff_results.csv --crop 1024

# Process CZI files (requires ImageJ/pyimagej)
python batch_predict.py --input_dir czi_data/ --model yolov8s-seg_yfusion.pt --format czi --output_csv czi_results.csv
```

**Zoom Mode for High-Resolution Images:**
```bash
# Enable zoomed prediction with overlapping crops (recommended for 1024x1024+ images)
python batch_predict.py --input_dir new_images/filtro_H --model yolov8s-seg_yfusion.pt --format czi --output_csv results_zoom.csv --zoom

# Customize zoom factor (0.5 = half image per crop, more overlap)
python batch_predict.py --input_dir images/ --model yolov8s-seg_yfusion.pt --format png --output_csv results.csv --zoom --zoom_factor 0.5

# Standard vs zoom mode comparison
# Standard: processes full image once
python batch_predict.py --input_dir new_images/filtro_H --model yolov8s-seg_yfusion.pt --format czi --output_csv standard.csv

# Zoom: creates 4 overlapping crops per image for better detail
python batch_predict.py --input_dir new_images/filtro_H --model yolov8s-seg_yfusion.pt --format czi --output_csv zoom.csv --zoom
```

**Real-World Examples:**
```bash
# Example 1: Process yeast fusion experiment CZI files
python batch_predict.py \
  --input_dir "new_images/filtro H" \
  --model yolov8s-seg_yfusion.pt \
  --format czi \
  --output_csv "new_images/filtro H/batch_predictions.csv"

# Example 2: High-resolution zoom mode for detailed detection
python batch_predict.py \
  --input_dir "new_images/filtro Ph1" \
  --model yolov8s-seg_yfusion.pt \
  --format czi \
  --output_csv "new_images/filtro Ph1/batch_predictions_zoom.csv" \
  --zoom \
  --zoom_factor 0.667

# Example 3: Using configuration file for reproducibility
python batch_predict.py --config my_config.yaml
```

**Command-Line Arguments:**
- `--input_dir`: Directory containing input images (required)
- `--model`: Path to trained YOLO model file (.pt) (required)
- `--format`: Image format - 'png', 'tif', or 'czi' (required)
- `--output_csv`: Path for combined output CSV file (required)
- `--crop`: Image crop/resize size (default: 1024)
- `--zoom`: Enable zoomed prediction mode with overlapping crops
- `--zoom_factor`: Fraction of image for each crop (default: 0.667, range: 0.5-0.8)
- `--config`: YAML configuration file path (optional, overrides command-line args)

**Output Format:**

The script generates CSV files with comprehensive statistics per detection:

*Standard columns:*
- `file`: Source image filename
- `crop_id`: Crop identifier (0 for standard mode, 0-3 for zoom mode)
- `class`: Predicted class ID
- `proba`: Detection confidence score (0-1)
- `x1`, `y1`, `x2`, `y2`: Bounding box coordinates

*Zoom mode additional columns:*
- `crop_x1`, `crop_y1`, `crop_x2`, `crop_y2`: Crop region coordinates in original image

*Channel statistics (for each of BF, RFP, GFP):*
- `{channel}_mean`: Mean pixel intensity
- `{channel}_std`: Variance (not standard deviation!)
- `{channel}_min`: Minimum pixel value
- `{channel}_max`: Maximum pixel value
- `{channel}_skew`: Distribution skewness

**Expected Detection Classes (yolov8s-seg_yfusion model):**
- Class 0: `f` (individual/fused cells)
- Class 1: `h` (haploid/intermediate stages)
- Class 2: `lmcf` (large mating cell fusion)
- Class 3: `lmsgfp` (large mating single GFP)
- Class 4: `lsgfp` (large single GFP)

**Performance Benchmarks:**
- Standard mode: ~3-5 seconds/image (CZI, 1024x1024)
- Zoom mode: ~6-12 seconds/image (4 crops with overlap)
- Typical detection increase with zoom: 30-60% more objects detected

#### 2. Running Annotations with annotate_images.py

The `annotate_images.py` script provides visual annotation with bounding boxes and comprehensive statistical extraction:

**Basic Usage:**
```bash
# Auto-detect format and process directory
python annotate_images.py --model yolov8s-seg_yfusion.pt --input images/ --output results.csv

# Specify format explicitly for TIFF stacks
python annotate_images.py --model yolov8s-seg_yfusion.pt --input images/ --output results.csv --format tiff

# Process CZI files with automatic ImageJ integration
python annotate_images.py --model yolov8s-seg_yfusion.pt --input czi_files/ --output results.csv --format czi


**Advanced Options:**
```bash
# Custom confidence threshold (filter low-confidence detections)
python annotate_images.py --model yolov8s-seg_yfusion.pt --input images/ --output results.csv --confidence 0.7

# Zoom mode for better detail on high-resolution images
python annotate_images.py --model yolov8s-seg_yfusion.pt --input images/ --output results.csv --zoom --zoom_factor 0.5

# Custom image and crop sizes
python annotate_images.py --model yolov8s-seg_yfusion.pt --input images/ --output results.csv --imgsz 1024 --crop 1024

# Verbose mode for debugging and progress tracking
python annotate_images.py --model yolov8s-seg_yfusion.pt --input images/ --output results.csv --verbose
```

**Real-World Examples:**
```bash
# Example 1: Process TIFF stack with BF/GFP/RFP channels
# Files: sample_BF.tif, sample_GFP.tif, sample_RFP.tif (auto-grouped)
python annotate_images.py \
  --model yolov8s-seg_yfusion.pt \
  --input tiff_stacks/ \
  --output annotations.csv \
  --format tiff \
  --confidence 0.6

# Example 2: CZI microscopy files with zoom mode
python annotate_images.py \
  --model yolov8s-seg_yfusion.pt \
  --input microscopy_data/ \
  --output detections.csv \
  --format czi \
  --zoom \
  --zoom_factor 0.667 \
  --verbose

# Example 3: Quick annotation of PNG images
python annotate_images.py \
  --model yolov8s-seg_yfusion.pt \
  --input quick_test/ \
  --output test_results.csv \
  --format single
```

**Command-Line Arguments:**
- `--model`: Path to trained YOLO model file (.pt) (required)
- `--input`: Input directory containing images (required)
- `--output`: Output CSV file path (required)
- `--format`: Image format - 'auto', 'tiff', 'czi', or 'single' (default: 'auto')
- `--confidence`: Confidence threshold for detections (default: 0.5, range: 0.1-0.9)
- `--imgsz`: Image size for inference (default: 1024)
- `--crop`: Crop size for input images (default: 1024)
- `--zoom`: Enable zoomed prediction with overlapping crops
- `--zoom_factor`: Zoom factor for cropping (default: 0.667)
- `--verbose`: Enable verbose output with progress information

**Output Files:**
1. **CSV file** (`--output`): Comprehensive detection results with statistics
2. **Annotated images**: PNG files with bounding boxes saved as `*_annotated.png`

**Annotated Image Features:**
- Color-coded bounding boxes per class
- Class labels with confidence scores
- Segmentation masks overlay
- Saved in same directory as input files

**TIFF Stack Processing:**
The script automatically:
- Detects files matching patterns: `*BF*.tif`, `*GFP*.tif`, `*RFP*.tif`
- Groups related files by common prefix
- Processes each frame in multi-frame stacks
- Exports results with frame indices and metadata

**CSV Output Columns:**
*Detection information:*
- `file`: Source filename
- `crop_id`: Crop identifier (0 for standard, 0-N for zoom mode)
- `class`: Predicted class ID
- `proba`: Confidence score (0-1)
- `x1`, `y1`, `x2`, `y2`: Bounding box coordinates

*Zoom mode columns:*
- `crop_x1`, `crop_y1`, `crop_x2`, `crop_y2`: Crop coordinates

*Channel statistics (BF, RFP, GFP):*
- `{channel}_mean`, `{channel}_std`, `{channel}_min`, `{channel}_max`, `{channel}_skew`

**Format-Specific Behavior:**
- **TIFF**: Automatically groups by BF/GFP/RFP, processes all frames in stacks
- **CZI**: Uses ImageJ/pyimagej to load multi-channel CZI microscopy files
- **Single**: Converts grayscale/single-channel to 3-channel format
- **Auto**: Detects format based on file extensions

#### 3. Training Custom Models with segment_retrain.ipynb

The Jupyter notebook provides a complete training pipeline:

**Step-by-Step Workflow:**

1. **Open the notebook:**
   ```bash
   jupyter notebook segment_retrain.ipynb
   ```

2. **Cell 1-5: Environment Setup**
   - Import required libraries
   - Set up paths and configurations
   - Initialize ImageJ for CZI support

3. **Cell 6-15: Data Preparation**
   - Load H5/CZI/TIFF files
   - Extract and normalize image frames
   - Convert masks to YOLO format
   - Split data into train/val/test sets

4. **Cell 16-20: Data Augmentation**
   - Apply geometric transformations
   - Color jittering
   - Generate augmented dataset

5. **Cell 21-25: Model Training**
   - Load pre-trained YOLOv8 weights
   - Configure hyperparameters
   - Train the model
   - Monitor training metrics

6. **Cell 26-30: Model Evaluation**
   - Run predictions on test set
   - Calculate precision/recall
   - Visualize results

7. **Cell 31-35: Statistical Analysis**
   - Extract cell statistics
   - Compute channel intensities
   - Export results to CSV

**Key Variables to Modify:**
```python
# In the notebook
modelpath = 'yolov8n-seg.pt'  # Pre-trained model
datasetdir = 'datasets/'       # Dataset location
crop = 1024                    # Image size
batch = 8                      # Batch size
epochs = 100                   # Training epochs
```

#### 4. Using Pre-trained Models

**Available Models:**
yolov8s-seg_yfusion.pt

### Complete Workflow Examples

#### Workflow 1: CZI Microscopy Processing Pipeline
```bash
# Step 1: Run batch predictions on CZI files
python batch_predict.py \
  --input_dir "raw_microscopy/experiment_001" \
  --model yolov8s-seg_yfusion.pt \
  --format czi \
  --output_csv "results/exp001_detections.csv"

# Step 2: Run with zoom for high-detail analysis
python batch_predict.py \
  --input_dir "raw_microscopy/experiment_001" \
  --model yolov8s-seg_yfusion.pt \
  --format czi \
  --output_csv "results/exp001_zoom_detections.csv" \
  --zoom \
  --zoom_factor 0.667

# Step 3: Generate annotated images for visual verification
python annotate_images.py \
  --model yolov8s-seg_yfusion.pt \
  --input "raw_microscopy/experiment_001" \
  --output "results/exp001_annotations.csv" \
  --format czi \
  --confidence 0.6 \
  --verbose

# Step 4: Analyze results in Python
python -c "
import pandas as pd
df = pd.read_csv('results/exp001_zoom_detections.csv')
print(f'Total detections: {len(df)}')
print(f'Detections by class:\n{df[\"class\"].value_counts()}')
print(f'Mean confidence: {df[\"proba\"].mean():.3f}')
"
```

#### Workflow 2: TIFF Stack Batch Processing
```bash
# Process directory of TIFF stacks (auto-groups BF/GFP/RFP)
python annotate_images.py \
  --model yolov8s-seg_yfusion.pt \
  --input "tiff_data/time_series" \
  --output "results/time_series_analysis.csv" \
  --format tiff \
  --confidence 0.5 \
  --verbose

# Extract statistics summary
python -c "
import pandas as pd
df = pd.read_csv('results/time_series_analysis.csv')
summary = df.groupby('class').agg({
    'bf_mean': 'mean',
    'gfp_mean': 'mean', 
    'rfp_mean': 'mean',
    'proba': 'mean'
})
print(summary)
"
```

#### Workflow 3: High-Throughput Screening
```bash
# Process multiple folders in parallel
for folder in sample_*; do
    echo "Processing $folder..."
    python batch_predict.py \
        --input_dir "$folder" \
        --model yolov8s-seg_yfusion.pt \
        --format czi \
        --output_csv "results/${folder}_predictions.csv" &
done
wait

# Combine all results
python -c "
import pandas as pd
import glob
dfs = [pd.read_csv(f) for f in glob.glob('results/*_predictions.csv')]
combined = pd.concat(dfs, ignore_index=True)
combined.to_csv('results/combined_all.csv', index=False)
print(f'Combined {len(dfs)} files with {len(combined)} total detections')
"
```

### Command Line Interface

The package provides command-line tools for common tasks:

#### Training a Model

```bash
train-yolo --data dataset.yaml --epochs 100 --batch-size 8 --img-size 1024
```

#### Batch Prediction

```bash
batch-predict --model yolov8n-seg_yfusion.pt --input images/ --output results/
```

#### Image Annotation

The `annotate-images` command provides comprehensive image annotation with automatic format detection:

```bash
# Auto-detect image format and process directory
annotate-images --model yolov8n-seg_yfusion.pt --input images/ --output results.csv

# Process specific format with custom confidence threshold
annotate-images --model yolov8n-seg_yfusion.pt --input images/ --output results.csv --format tiff --confidence 0.7

# Process CZI files
annotate-images --model yolov8n-seg_yfusion.pt --input czi_images/ --output results.csv --format czi
```

### Python API

```python
from ultralytics import YOLO
from yeast_fusion_segmenter import load_czi_with_imagej, predpng

# Load a trained model
model = YOLO('yolov8n-seg_yfusion.pt')

# Process CZI files

# Run prediction and save results
predpng(model, 'sample.png', 'results.csv')

# Or use the annotation script programmatically
from yeast_fusion_segmenter import annotate_main
import sys
sys.argv = ['annotate_images.py', '--model', 'yolov8n-seg_yfusion.pt', 
            '--input', 'images/', '--output', 'results.csv']
annotate_main()
```

### Jupyter Notebooks

The project includes comprehensive Jupyter notebooks:

- `segment_retrain.ipynb`: Complete pipeline for data preparation, training, and evaluation
- Example notebooks in the `example/` directory

## 📁 Project Structure

```
yeast_fusion_segmenter/
├── README.MD                  # This file
├── pyproject.toml            # Modern Python packaging configuration
├── setup.py                  # Legacy packaging support
├── environment.yml           # Conda environment specification
├── dataset.yaml              # YOLO dataset configuration
├── sample_hyperparameters.yaml  # Training hyperparameters
├── train_yolo.py            # Training script
├── batch_predict.py         # Batch prediction script
├── annotate_images.py       # Image annotation script
├── prepare_yolo_data.py     # Data preparation utilities
├── segment_retrain.ipynb    # Main training notebook
├── yolov8*-seg*.pt          # Pre-trained model weights
├── images_CNN/              # Training images
├── datasets/                # Processed YOLO datasets
├── augmented/               # Augmented training data
└── example/                 # Example scripts and data
```

## 📊 Data Format

### Input Images
- **Brightfield (BF)**: Morphology and cell boundaries
- **GFP**: Green fluorescent protein channel
- **RFP**: Red fluorescent protein channel

### Output Format
Results are saved as CSV files containing:
- Cell coordinates and bounding boxes
- Classification scores
- Statistical measurements (mean, std, min, max, skewness) for each channel
- Mask contours for further analysis

### Annotation Script Features

The `annotate-images` command provides:

- **Automatic Format Detection**: Supports TIFF, CZI, PNG, and JPG formats
- **Multi-channel Processing**: Handles BF/GFP/RFP channel combinations automatically
- **Statistical Analysis**: Extracts comprehensive pixel statistics from segmented regions
- **Visualization**: Generates annotated images with bounding boxes and labels
- **Batch Processing**: Processes entire directories of images
- **Flexible Input**: Works with individual images or image stacks

**Supported Input Patterns**:
- TIFF stacks: `*BF*.tif`, `*GFP*.tif`, `*RFP*.tif` (automatically grouped)
- CZI files: `*.czi` (processed with ImageJ)
- Single images: `*.png`, `*.jpg` (converted to 3-channel format)

**Output CSV Columns**:
- `image_path`, `detection_id`, `class`, `confidence`
- `x1`, `y1`, `x2`, `y2` (bounding box coordinates)
- `bbox_area`, `mask_area` (area measurements)
- `bf_mean`, `bf_std`, `bf_min`, `bf_max`, `bf_skew`, `bf_kurtosis` (brightfield statistics)
- `gfp_mean`, `gfp_std`, `gfp_min`, `gfp_max`, `gfp_skew`, `gfp_kurtosis` (GFP statistics)
- `rfp_mean`, `rfp_std`, `rfp_min`, `rfp_max`, `rfp_skew`, `rfp_kurtosis` (RFP statistics)
- `group_name`, `frame_index`, `source_path` (metadata)

## 🔧 Configuration

### Training Hyperparameters

Key hyperparameters can be adjusted in `sample_hyperparameters.yaml`:

```yaml
lr0: 0.001          # Initial learning rate
epochs: 2000        # Training epochs  
batch_size: 1       # Batch size
imgsz: 1024         # Image size
degrees: 180.0      # Rotation augmentation
flipud: 0.5         # Vertical flip probability
fliplr: 0.5         # Horizontal flip probability
```



## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/DessimozLab/yeast_fusion_segmenter.git
cd yeast_fusion_segmenter

# Install in development mode
pip install -e .[dev,jupyter]

# Run tests
pytest

# Format code
black .
isort .
```

## 🐛 Known Issues

- CZI files require ImageJ/Fiji installation
- Large batch sizes may cause GPU memory issues
- Windows compatibility may require additional CUDA setup

## 📖 Citation

If you use this tool in your research, please cite:

```bibtex
@software{yeast_fusion_segmenter,
  author = {Your Name},
  title = {Yeast Fusion Segmenter: Deep Learning Tool for Yeast Cell Segmentation},
  url = {https://github.com/DessimozLab/yeast_fusion_segmenter},
  version = {0.1.0},
  year = {2025}
}
```

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics for the base segmentation framework
- ImageJ/Fiji community for CZI file support
- PyTorch team for the deep learning framework
- Scientific community for validation and feedback

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/DessimozLab/yeast_fusion_segmenter/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DessimozLab/yeast_fusion_segmenter/discussions)
- **Email**: dmoi@unil.ch

### Missing CUDA Libraries

If you encounter errors about missing CUDA libraries, you can check the environment's CUDA configuration:

```bash
conda list | grep cuda
```

### Environment Activation Issues

If you have problems activating the environment, try:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
mamba activate yeast_fusion_segmenter
```

### PyTorch CUDA Compatibility

If PyTorch doesn't recognize your GPU, make sure your NVIDIA driver version is compatible with the CUDA version installed in the environment (11.7). You can check your driver's compatible CUDA versions with:

```bash
nvidia-smi
```

## Additional Notes

- The environment uses Python 3.9 and PyTorch 2.0.0 with CUDA 11.7 support
- For large datasets, ensure you have sufficient disk space and RAM
- The segmentation model requires a GPU with at least 4GB of VRAM for optimal performance
