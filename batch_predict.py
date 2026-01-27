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
import numpy as np
import pandas as pd
from PIL import Image, ImageSequence
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
        # Initialize ImageJ (headless)
        ij = imagej.init('sc.fiji:fiji', headless=True)
        dataset = ij.io().open(path)
        img5d = ij.py.from_java(dataset)
        # img5d shape: (T, C, Z, Y, X) or similar
        # Take first timepoint, first channel, first Z
        if img5d.ndim == 5:
            plane = img5d[0, 0, 0]
        elif img5d.ndim == 4:
            plane = img5d[0, 0]
        else:
            plane = img5d.squeeze()
        plane = np.array(plane)
        
        # Normalize to 0-255 uint8
        plane = plane.astype(np.float32)
        plane -= plane.min()
        if plane.max() > 0:
            plane /= plane.max()
        plane_uint8 = (plane * 255).astype(np.uint8)
        
        # Convert to PIL image
        pil_img = Image.fromarray(plane_uint8)
        
        # Resize to crop size
        pil_img = pil_img.resize((crop, crop))
        
        # Convert to RGB
        rgb_img = pil_img.convert('RGB')
        
        # Save as PNG for YOLO prediction
        png_path = path.replace('.czi', '.png').replace('.CZI', '.png')
        rgb_img.save(png_path)
        
        # Return array and PNG path
        return np.array(rgb_img).astype(np.uint8), png_path
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
    
    # Validate required arguments
    required = ['input_dir', 'model', 'format', 'output_csv']
    missing = [arg for arg in required if not getattr(args, arg, None)]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}. Provide via --config or command line.")

    model = YOLO(args.model)
    pattern = {'png': '*.png', 'tif': '*.tif', 'czi': '*.czi'}[args.format]
    files = sorted(glob.glob(os.path.join(args.input_dir, pattern)))
    all_dfs = []
    
    for imgfile in tqdm.tqdm(files):
        print(f"Processing {imgfile}...")
        # Load image as array and get PNG path (for CZI)
        arr, png_path = process_image(imgfile, args.format, crop=args.crop)
        
        if args.zoom:
            # Zoomed prediction with multiple crops
            crops, png_paths, coordinates = zoom_img(arr, args.zoom_factor, args.crop, base_filepath=imgfile)
            print(f"  Created {len(crops)} zoomed crops with {len(png_paths)} PNG files")
            
            for crop_idx, (crop, crop_png_path, coord) in enumerate(zip(crops, png_paths, coordinates)):
                # Use the saved PNG path directly
                df = predict_and_collect(model, crop_png_path,
                                       crop_png_path.replace('.png', '.csv'),
                                       crop=args.crop, crop_id=crop_idx)
                if df is not None:
                    # Add coordinate information
                    df['crop_x1'] = coord[0]
                    df['crop_y1'] = coord[1]
                    df['crop_x2'] = coord[2]
                    df['crop_y2'] = coord[3]
                    all_dfs.append(df)
                # Clean up the crop PNG file
                if os.path.exists(crop_png_path):
                    os.remove(crop_png_path)
        else:
            # Normal prediction on full image
            # Use PNG path directly for CZI files, create temp for others
            if png_path is not None:
                # CZI file - use the saved PNG directly
                pred_path = png_path
                df = predict_and_collect(model, pred_path,
                                       pred_path.replace('.png', '.csv'),
                                       crop=args.crop, crop_id=0)
                if df is not None:
                    all_dfs.append(df)
            else:
                # TIFF or PNG - create temp file
                temp_path = imgfile + '_yoloinput.png'
                Image.fromarray(arr.astype(np.uint8)).save(temp_path)
                df = predict_and_collect(model, temp_path,
                                       temp_path.replace('.png', '.csv'),
                                       crop=args.crop, crop_id=0)
                if df is not None:
                    all_dfs.append(df)
                os.remove(temp_path)
    
    if all_dfs:
        pd.concat(all_dfs).to_csv(args.output_csv, index=False)
        print(f'Wrote results to {args.output_csv}')
    else:
        print('No predictions made.')

if __name__ == '__main__':
    main()
