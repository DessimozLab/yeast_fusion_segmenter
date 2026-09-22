#!/usr/bin/env python3
"""
Batch Prediction Script for Yeast Fusion Segmenter

This script performs batch prediction on multiple images using a trained YOLOv8
segmentation model. It supports multiple image formats (PNG, TIFF, CZI) and can
process images either as full frames or with overlapping crops (zoom mode) for
large high-resolution images.

Key Features:
- Multi-format support (PNG, TIFF, CZI)
- Zoom mode for processing large images with overlapping crops
- Statistical analysis of segmented regions (mean, variance, min, max, skewness)
- Exports results to CSV with bounding boxes, masks, and channel statistics

Author: Yeast Fusion Segmenter Team
License: MIT
"""

import argparse
import os
import glob
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageSequence
from pathlib import Path
from ultralytics import YOLO
from scipy.stats import describe
import cv2
import tqdm
import yaml

# Optional: for CZI support - requires ImageJ/Fiji installation
try:
    import imagej
except ImportError:
    imagej = None


_IJ = None


def _get_imagej_instance():
    """Lazily initialize a single headless ImageJ runtime."""
    global _IJ
    if _IJ is None:
        _IJ = imagej.init('sc.fiji:fiji', mode='headless')
    return _IJ


def percentile_u8(channel, low_pct=1.0, high_pct=99.0):
    """Robustly scale one channel to uint8 using percentile clipping."""
    ch = channel.astype(np.float32)
    lo, hi = np.percentile(ch, [low_pct, high_pct])
    if hi <= lo:
        lo, hi = float(ch.min()), float(ch.max())
        if hi <= lo:
            return np.zeros_like(ch, dtype=np.uint8)
    ch = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    return (ch * 255).astype(np.uint8)


def convert_czi_to_rgb_uint8(raw_array):
    """Convert raw CZI array to RGB uint8 with robust channel handling."""
    arr = np.asarray(raw_array)
    arr = np.squeeze(arr)

    while arr.ndim > 3:
        arr = arr[0]

    if arr.ndim == 2:
        g = percentile_u8(arr)
        return np.stack([g, g, g], axis=-1)

    if arr.ndim != 3:
        raise ValueError(f'Unsupported CZI array shape after squeeze: {arr.shape}')

    channel_axis = None
    for i, s in enumerate(arr.shape):
        if s in (3, 4):
            channel_axis = i
            break
    if channel_axis is None:
        channel_axis = int(np.argmin(arr.shape))

    if channel_axis != 2:
        arr = np.moveaxis(arr, channel_axis, 2)

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] == 2:
        arr = np.concatenate(
            [arr, np.zeros((arr.shape[0], arr.shape[1], 1), dtype=arr.dtype)],
            axis=2,
        )
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]

    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    for c in range(3):
        rgb[:, :, c] = percentile_u8(arr[:, :, c])
    return rgb


def center_crop_or_pad(image, size=1024):
    """Center crop to size and zero-pad if source image is smaller."""
    h, w = image.shape[:2]
    y0 = max((h - size) // 2, 0)
    x0 = max((w - size) // 2, 0)
    y1 = min(y0 + size, h)
    x1 = min(x0 + size, w)
    cropped = image[y0:y1, x0:x1, ...]

    if image.ndim == 2:
        out = np.zeros((size, size), dtype=image.dtype)
        out[:cropped.shape[0], :cropped.shape[1]] = cropped
    else:
        out = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
        out[:cropped.shape[0], :cropped.shape[1], :] = cropped
    return out


def yield_frames(img, crop=1024, verbose=False, scaler=True):
    """
    Extract and normalize frames from multi-page TIFF images.
    
    This generator function iterates through all pages in a multi-frame TIFF file,
    crops them to the specified size, and applies normalization if requested.
    
    Args:
        img (PIL.Image): Multi-page TIFF image object
        crop (int): Size to crop each frame to (default: 1024)
        verbose (bool): Print debug information (default: False)
        scaler (bool): Apply min-max normalization to 0-255 range (default: True)
        
    Yields:
        numpy.ndarray: Processed frame as uint8 array with shape (crop, crop)
    """
    for i, page in enumerate(ImageSequence.Iterator(img)):
        page_array = np.array(page)
        if crop is not None:
            page_array = page_array[0:crop, 0:crop]
        if scaler:
            if page_array.max() > page_array.min():
                page_array = (page_array - page_array.min()) / (page_array.max() - page_array.min()) * 255
            else:
                page_array = np.zeros_like(page_array)
        yield page_array.astype(np.uint8)

def process_image(path, fmt, crop=1024):
    """Process image and return both array and PNG path for CZI files.
    
    Returns:
        tuple: (image_array, png_path) for CZI, (image_array, None) for others
    """
    if fmt == 'czi':
        if imagej is None:
            raise ImportError('imagej is required for CZI format. Install with `pip install imagej`')
        ij = _get_imagej_instance()
        dataset = ij.io().open(path)
        raw = ij.py.from_java(dataset)
        rgb_img = center_crop_or_pad(convert_czi_to_rgb_uint8(raw), size=crop)
        
        # Save as PNG for YOLO prediction
        png_path = path.replace('.czi', '.png').replace('.CZI', '.png')
        Image.fromarray(rgb_img).save(png_path)
        
        # Return array and PNG path
        return rgb_img, png_path
    elif fmt == 'tif':
        img = Image.open(path)
        frames = [frame for frame in yield_frames(img, crop=crop)]
        if len(frames) == 1:
            img = np.stack([frames[0]]*3, axis=-1)
        elif len(frames) >= 3:
            img = np.stack(frames[:3], axis=-1)
        else:
            img = np.stack([frames[0]]*3, axis=-1)
        return img, None
    elif fmt == 'png':
        img = Image.open(path).convert('RGB')
        img = img.resize((crop, crop))
        return np.array(img), None
    else:
        raise ValueError(f'Unknown format: {fmt}')


def zoom_img(img, zoom_factor, target_size=1024, base_filepath=None):
    """
    Create overlapping crops from a large image for zoomed predictions.
    
    Args:
        img: Image array (H, W, C)
        zoom_factor: Fraction of image to use for each crop (e.g., 40/60)
        target_size: Size to resize each crop to
        base_filepath: Base filepath for saving PNG crops (optional)
        
    Returns:
        tuple: (crops, png_paths, coordinates)
            - crops: List of cropped image arrays
            - png_paths: List of paths to saved PNG files
            - coordinates: List of (x1, y1, x2, y2) tuples
    """
    y_size = int(img.shape[0] * zoom_factor)
    x_size = int(img.shape[1] * zoom_factor)
    crops = []
    png_paths = []
    coordinates = []
    crop_number = 0
    
    for y in range(0, img.shape[0] - y_size + 1, y_size // 2):
        for x in range(0, img.shape[1] - x_size + 1, x_size // 2):
            sub = img[y:y + y_size, x:x + x_size]
            # Interpolate to target size
            sub_resized = cv2.resize(sub, (target_size, target_size),
                                   interpolation=cv2.INTER_CUBIC)
            
            # Save as PNG if base_filepath provided
            if base_filepath:
                # Convert to uint8 for saving
                if sub_resized.dtype == np.float32 or sub_resized.dtype == np.float64:
                    sub_uint8 = (sub_resized * 255).astype(np.uint8)
                else:
                    sub_uint8 = sub_resized
                
                # Create PNG filename based on original file and crop number
                base_name = os.path.splitext(base_filepath)[0]
                png_path = f"{base_name}_crop_{crop_number}.png"
                Image.fromarray(sub_uint8).save(png_path)
                png_paths.append(png_path)
            
            crops.append(sub_resized)
            coordinates.append((x, y, x + x_size, y + y_size))
            crop_number += 1
    
    return crops, png_paths, coordinates


def predict_and_collect(model, imgfile, outcsv, crop=1024, crop_id=None):
    """
    Run YOLO prediction on an image and collect statistical results.
    
    This function performs instance segmentation on an image using a trained YOLO model,
    extracts pixel-level statistics from each detected cell, and saves results to CSV.
    For each detected object, it computes statistics for all three channels (BF, RFP, GFP).
    
    Args:
        model (YOLO): Trained YOLOv8 segmentation model
        imgfile (str): Path to the input image file
        outcsv (str): Path to save the output CSV file
        crop (int): Image size for model input (default: 1024)
        crop_id (int, optional): Crop identifier for zoom mode (default: None)
        
    Returns:
        pandas.DataFrame: Results dataframe with columns for bounding boxes, class,
                         confidence, and channel statistics (BF/RFP/GFP mean, std,
                         min, max, skewness), or None if no detections
    
    Note:
        - Only detections with confidence > 0.5 are included
        - Statistics are computed from pixels within the segmentation mask
        - Channel order is assumed to be BF (0), RFP (1), GFP (2)
    """
    # Run YOLO model inference on the image
    results = model(imgfile, imgsz=crop, visualize=False)
    
    # Extract detection information from results
    classes = results[0].names  # Class names dictionary
    nboxes = len(results[0].boxes)  # Number of detected objects
    proba = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    xywh = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
    masks = results[0].masks  # Segmentation masks
    
    # Load the original image for pixel extraction
    img = np.array(Image.open(imgfile))
    
    # Dictionary to store results for each detection
    resdict = {}
    rescount = 0  # Counter for successful detections
    
    # Iterate through all detected objects
    for i in range(nboxes):
        # Get class ID for this detection
        c = int(results[0].boxes.cls[i])
        
        # Only process detections above confidence threshold
        if proba[i] > 0.5:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = xywh[i]
            c = int(c)
            
            # Initialize result dictionary for this detection
            resdict[rescount] = {
                'file': os.path.basename(imgfile),
                'crop_id': crop_id if crop_id is not None else 0,
                'class': c,
                'proba': proba[i],
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            }
            
            # Process mask if available
            if masks is not None:
                # Get mask polygon coordinates
                mask = masks[i].xy[0]  # Shape: (N, 2) array of (x, y) coordinates
                mask = mask.astype(int)
                
                # Extract pixel values for each channel using mask coordinates
                # mask[:, 1] are y-coordinates, mask[:, 0] are x-coordinates
                bf = img[mask[:, 1], mask[:, 0], 0].ravel()   # Brightfield channel
                rfp = img[mask[:, 1], mask[:, 0], 1].ravel()  # RFP channel
                gfp = img[mask[:, 1], mask[:, 0], 2].ravel()  # GFP channel
                
                # Calculate statistical descriptors for each channel
                bf_stats = describe(bf)
                rfp_stats = describe(rfp)
                gfp_stats = describe(gfp)
                
                # Create statistics dictionaries for each channel
                # Note: Using variance instead of std (can compute std = sqrt(variance))
                bf_stats_dict = {
                    'bf_mean': bf_stats.mean,           # Mean intensity
                    'bf_std': bf_stats.variance,         # Variance (not std!)
                    'bf_min': bf_stats.minmax[0],       # Minimum intensity
                    'bf_max': bf_stats.minmax[1],       # Maximum intensity
                    'bf_skew': bf_stats.skewness        # Distribution skewness
                }
                rfp_stats_dict = {
                    'rfp_mean': rfp_stats.mean,
                    'rfp_std': rfp_stats.variance,
                    'rfp_min': rfp_stats.minmax[0],
                    'rfp_max': rfp_stats.minmax[1],
                    'rfp_skew': rfp_stats.skewness
                }
                gfp_stats_dict = {
                    'gfp_mean': gfp_stats.mean,
                    'gfp_std': gfp_stats.variance,
                    'gfp_min': gfp_stats.minmax[0],
                    'gfp_max': gfp_stats.minmax[1],
                    'gfp_skew': gfp_stats.skewness
                }

                resdict[rescount].update(bf_stats_dict)
                resdict[rescount].update(rfp_stats_dict)
                resdict[rescount].update(gfp_stats_dict)
            else:
                # If no mask, set statistics to NaN
                resdict[rescount].update({
                    'bf_mean': np.nan,
                    'bf_std': np.nan,
                    'bf_min': np.nan,
                    'bf_max': np.nan,
                    'bf_skew': np.nan,
                    'rfp_mean': np.nan,
                    'rfp_std': np.nan,
                    'rfp_min': np.nan,
                    'rfp_max': np.nan,
                    'rfp_skew': np.nan,
                    'gfp_mean': np.nan,
                    'gfp_std': np.nan,
                    'gfp_min': np.nan,
                    'gfp_max': np.nan,
                    'gfp_skew': np.nan
                })
            
            rescount += 1

    if resdict:
        df = pd.DataFrame.from_dict(resdict, orient='index')
        print(df)
        print(len(df), 'objects detected')
        df.to_csv(outcsv)
        return df
    return None

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def dataset_prediction_files(dataset_info_path, split='all', annotated_only=False):
    """Return prepared PNGs and provenance from a saved dataset object.

    Dataset-info JSON is the durable interface between preparation, training,
    and inference.  It points to the materialized YOLO PNGs and contains the
    original CZI/TIFF paths plus the deterministic HDF5 pairing.  Inference
    therefore does not need to reopen CZI files or rediscover raw files.
    """
    from image_dataset import MicroscopyImageDataset

    info_path = Path(dataset_info_path).resolve()
    info = MicroscopyImageDataset.load_dataset_info(info_path)
    yolo_data = info.get('yolo_data')
    if not yolo_data:
        raise ValueError(f"Dataset info has no prepared YOLO data path: {info_path}")
    yolo_yaml = Path(yolo_data).resolve()
    if not yolo_yaml.exists():
        raise FileNotFoundError(f"Prepared YOLO dataset YAML not found: {yolo_yaml}")

    config = yaml.safe_load(yolo_yaml.read_text()) or {}
    dataset_root = Path(config.get('path', yolo_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (yolo_yaml.parent / dataset_root).resolve()

    selected_splits = ('train', 'val', 'test') if split == 'all' else (split,)
    record_lookup = {
        (record['magnification'], record['sample_id']): record
        for record in info['records']
    }
    files = []
    for selected_split in selected_splits:
        image_dir = dataset_root / selected_split / 'images'
        label_dir = dataset_root / selected_split / 'labels'
        if not image_dir.exists():
            raise FileNotFoundError(f"Dataset split has no image directory: {image_dir}")
        for image_path in sorted(image_dir.glob('*.png')):
            label_path = label_dir / f'{image_path.stem}.txt'
            if annotated_only and (not label_path.exists() or not label_path.read_text().strip()):
                continue
            try:
                magnification, materialized_id = image_path.stem.split('_', 1)
            except ValueError as error:
                raise ValueError(f"Invalid prepared image name: {image_path.name}") from error
            sample_id = materialized_id.split('__f', 1)[0]
            record = record_lookup.get((magnification, sample_id))
            if record is None:
                raise ValueError(
                    f"Prepared image {image_path.name} has no matching record in {info_path}"
                )
            metadata = {
                'dataset_name': info['name'],
                'dataset_split': selected_split,
                'sample_id': sample_id,
                'magnification': magnification,
                'source_format': record['source_format'],
                'source_paths': json.dumps(record['sources'], sort_keys=True),
                'annotation_path': record.get('annotation_path'),
            }
            files.append((image_path, metadata))
    if not files:
        qualifier = 'annotated ' if annotated_only else ''
        raise ValueError(f"No {qualifier}PNG images found for split '{split}' in {info_path}")
    return files


def append_metadata(dataframe, metadata):
    """Attach dataset provenance to each detected instance, when requested."""
    if dataframe is not None and metadata:
        for key, value in metadata.items():
            dataframe[key] = value
    return dataframe

def main():
    parser = argparse.ArgumentParser(
        description='Batch predict with YOLO and compile results to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process PNG images from test dataset
  python batch_predict.py --input_dir datasets/test/images --model yolov8n-seg_yfusion.pt --format png --output_csv results.csv
  
  # Process TIFF images with custom crop size
  python batch_predict.py --input_dir new_images --model yolov8n-seg_yfusionmk3.pt --format tif --output_csv predictions.csv --crop 512
  
  # Process CZI files (requires imagej)
  python batch_predict.py --input_dir raw_data --model yolov8n-seg_yfusion.pt --format czi --output_csv czi_results.csv
  
  # Process images with zoomed prediction (overlapping crops)
  python batch_predict.py --input_dir datasets/val/images --model yolov8n-seg_yfusionmk2.pt --format png --output_csv validation_results.csv --zoom
  
  # Process with custom zoom factor
  python batch_predict.py --input_dir images/ --model yolov8n-seg_yfusion.pt --format png --output_csv results.csv --zoom --zoom_factor 0.5

  # Predict a prepared dataset object's held-out test split
  python batch_predict.py --dataset data/dataset_info/all_images.json --split test \\
      --model models/all_images_yolov8n_seg.pt --output_csv test_predictions.csv

Note:
  - The script will create individual CSV files for each image and a combined output CSV
  - Supported formats: png, tif, czi
  - For CZI support, install imagej: pip install pyimagej
  - Results include bounding boxes, class predictions, confidence scores, and channel statistics (BF, RFP, GFP)
  - Use --zoom for overlapping crop predictions on large images (adds crop_id and crop coordinates to output)
  - Use --config to load all parameters from a YAML configuration file
        """)
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--input_dir', help='Directory with images')
    parser.add_argument('--dataset', help='Dataset-info JSON saved by MicroscopyImageDataset')
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='all',
                        help='Prepared dataset split to predict (default: all)')
    parser.add_argument('--annotated-only', action='store_true',
                        help='With --dataset, keep only PNGs with non-empty YOLO labels')
    parser.add_argument('--model', help='Path to YOLO model')
    parser.add_argument('--format', choices=['png', 'tif', 'czi'], help='Image format')
    parser.add_argument('--output_csv', help='Output CSV file')
    parser.add_argument('--crop', type=int, default=1024, help='Crop size (default: 1024)')
    parser.add_argument('--zoom', action='store_true', help='Use zoomed prediction with overlapping crops')
    parser.add_argument('--zoom_factor', type=float, default=40/60, help='Zoom factor (default: 0.667)')
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        # Override with command-line arguments if provided
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # A saved dataset object supplies model-ready PNGs and their provenance;
    # directory mode retains the original multi-format behaviour.
    if args.dataset and args.input_dir:
        parser.error('Use either --dataset or --input_dir, not both.')
    if args.annotated_only and not args.dataset:
        parser.error('--annotated-only requires --dataset.')
    required = ['model', 'output_csv']
    if not args.dataset:
        required.extend(['input_dir', 'format'])
    missing = [arg for arg in required if not getattr(args, arg, None)]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}. Provide via --config or command line.")

    model = YOLO(args.model)
    if args.dataset:
        files = dataset_prediction_files(args.dataset, args.split, args.annotated_only)
        per_image_dir = Path(args.output_csv).with_suffix('')
        per_image_dir.mkdir(parents=True, exist_ok=True)
    else:
        pattern = {'png': '*.png', 'tif': '*.tif', 'czi': '*.czi'}[args.format]
        files = [(Path(path), None) for path in sorted(glob.glob(os.path.join(args.input_dir, pattern)))]
        per_image_dir = None
    all_dfs = []
    
    for image_path, metadata in tqdm.tqdm(files):
        imgfile = str(image_path)
        print(f"Processing {imgfile}...")
        # Load image as array and get PNG path (for CZI)
        # Dataset objects reference already materialized, model-ready PNGs.
        image_format = 'png' if args.dataset else args.format
        arr, png_path = process_image(imgfile, image_format, crop=args.crop)
        
        if args.zoom:
            # Zoomed prediction with multiple crops
            crops, png_paths, coordinates = zoom_img(arr, args.zoom_factor, args.crop, base_filepath=imgfile)
            print(f"  Created {len(crops)} zoomed crops with {len(png_paths)} PNG files")
            
            for crop_idx, (crop, crop_png_path, coord) in enumerate(zip(crops, png_paths, coordinates)):
                # Use the saved PNG path directly
                outcsv = (per_image_dir / f'{image_path.stem}_crop_{crop_idx}.csv'
                          if per_image_dir else Path(crop_png_path).with_suffix('.csv'))
                df = predict_and_collect(model, crop_png_path, str(outcsv),
                                       crop=args.crop, crop_id=crop_idx)
                if df is not None:
                    # Add coordinate information
                    df['crop_x1'] = coord[0]
                    df['crop_y1'] = coord[1]
                    df['crop_x2'] = coord[2]
                    df['crop_y2'] = coord[3]
                    all_dfs.append(append_metadata(df, metadata))
                # Clean up the crop PNG file
                if os.path.exists(crop_png_path):
                    os.remove(crop_png_path)
        else:
            # Normal prediction on full image
            # Use PNG path directly for CZI files, create temp for others
            if png_path is not None:
                # CZI file - use the saved PNG directly
                pred_path = png_path
                outcsv = (per_image_dir / f'{image_path.stem}.csv'
                          if per_image_dir else Path(pred_path).with_suffix('.csv'))
                df = predict_and_collect(model, pred_path, str(outcsv),
                                       crop=args.crop, crop_id=0)
                if df is not None:
                    all_dfs.append(append_metadata(df, metadata))
            else:
                # TIFF or PNG - create temp file
                temp_path = imgfile + '_yoloinput.png'
                Image.fromarray(arr.astype(np.uint8)).save(temp_path)
                outcsv = (per_image_dir / f'{image_path.stem}.csv'
                          if per_image_dir else Path(temp_path).with_suffix('.csv'))
                df = predict_and_collect(model, temp_path, str(outcsv),
                                       crop=args.crop, crop_id=0)
                if df is not None:
                    all_dfs.append(append_metadata(df, metadata))
                os.remove(temp_path)
    
    if all_dfs:
        pd.concat(all_dfs).to_csv(args.output_csv, index=False)
        print(f'Wrote results to {args.output_csv}')
    else:
        print('No predictions made.')

if __name__ == '__main__':
    main()
