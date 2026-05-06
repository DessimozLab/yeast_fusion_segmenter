#!/usr/bin/env python3
"""
Image Annotation Script for Yeast Fusion Segmenter

This script processes images (TIFF, CZI) using a trained YOLOv8 model and outputs
segmentation results in CSV format with statistical analysis.

Based on the logic from segment_retrain.ipynb notebook.
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence
from ultralytics import YOLO
from scipy.stats import describe
import cv2
import tqdm
from pathlib import Path
import colour as clr
import yaml

# Optional import for CZI support
try:
    import imagej
    CZI_SUPPORT = True
except ImportError:
    imagej = None
    CZI_SUPPORT = False
    print("Warning: CZI support not available. Install with: pip install pyimagej")


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
    Extract frames from multi-frame TIFF images.
    
    Args:
        img: PIL Image object
        crop: Crop size (default: 1024)
        verbose: Print debug info
        scaler: Apply normalization scaling
        
    Yields:
        numpy array: Processed frame
    """
    for i, page in enumerate(ImageSequence.Iterator(img)):
        if verbose:
            print(f"Processing frame {i}")
        
        page_array = np.array(page)
        if crop is not None:
            page_array = page_array[0:crop, 0:crop]
        
        if scaler:
            if page_array.max() > page_array.min():
                page_array = (page_array - page_array.min()) / (page_array.max() - page_array.min()) * 255
            else:
                page_array = np.zeros_like(page_array)
        
        yield page_array.astype(np.uint8)


def load_czi_with_imagej(filepath, resize=None):
    """
    Load a CZI file using ImageJ and convert it to a numpy array.
    Also saves a PNG version for YOLO predictions.
    
    Args:
        filepath (str): Path to the CZI file
        resize (tuple): Optional tuple of (width, height) to resize the image
        
    Returns:
        tuple: (np.array normalized image array, str PNG file path)
    """
    if not CZI_SUPPORT:
        raise ImportError("CZI support not available. Install with: pip install pyimagej")
    
    ij = _get_imagej_instance()
    dataset = ij.io().open(filepath)
    raw = ij.py.from_java(dataset)

    rgb_uint8 = convert_czi_to_rgb_uint8(raw)

    if resize is not None:
        if isinstance(resize, tuple):
            if len(resize) == 2 and resize[0] == resize[1]:
                rgb_uint8 = center_crop_or_pad(rgb_uint8, int(resize[0]))
            else:
                rgb_uint8 = cv2.resize(rgb_uint8, resize, interpolation=cv2.INTER_CUBIC)
        else:
            rgb_uint8 = center_crop_or_pad(rgb_uint8, int(resize))

    png_path = filepath.replace('.czi', '.png').replace('.CZI', '.png')
    Image.fromarray(rgb_uint8).save(png_path)

    return rgb_uint8, png_path


def process_tiff_stack(bf_path, gfp_path, rfp_path, crop=1024, verbose=False):
    """
    Process a stack of TIFF images (BF, GFP, RFP) and combine them.
    
    Args:
        bf_path: Path to brightfield TIFF
        gfp_path: Path to GFP TIFF  
        rfp_path: Path to RFP TIFF
        crop: Crop size
        verbose: Debug output
        
    Returns:
        List of stacked RGB images
    """
    print(f"Loading BF: {bf_path}")
    bf_img = Image.open(bf_path)
    bf_frames = list(yield_frames(bf_img, crop=crop, scaler=True, verbose=verbose))
    
    print(f"Loading GFP: {gfp_path}")
    gfp_img = Image.open(gfp_path)
    gfp_frames = list(yield_frames(gfp_img, crop=crop, scaler=True, verbose=verbose))
    
    print(f"Loading RFP: {rfp_path}")
    rfp_img = Image.open(rfp_path)
    rfp_frames = list(yield_frames(rfp_img, crop=crop, scaler=True, verbose=verbose))
    
    # Ensure all have the same number of frames
    min_frames = min(len(bf_frames), len(gfp_frames), len(rfp_frames))
    
    stacked_images = []
    for i in range(min_frames):
        # Stack the three channels
        stacked = np.stack([bf_frames[i], gfp_frames[i], rfp_frames[i]], axis=-1)
        stacked_images.append(stacked)
    
    return stacked_images


def zoom_img(img, zoom_factor, target_size=1024, base_filepath=None):
    """
    Create overlapping crops from a large image for zoomed predictions.
    
    This function divides a large image into overlapping sub-regions (crops),
    enabling detailed analysis of high-resolution images by the segmentation model.
    Each crop is resized to the target size while maintaining visual quality.
    
    The overlapping strategy uses 50% overlap (stride = crop_size // 2) to
    ensure no cellular features are missed at crop boundaries.
    
    Args:
        img (numpy.ndarray): Input image array with shape (H, W, C)
        zoom_factor (float): Fraction of image to use for each crop.
                           E.g., 0.667 means each crop covers 2/3 of image dimension.
                           Smaller values create more crops with better detail.
        target_size (int): Size to resize each crop to (default: 1024)
        base_filepath (str): Base filepath for saving PNG crops (optional)
        
    Returns:
        tuple: Contains three lists:
            - crops (list): List of cropped and resized image arrays
            - png_paths (list): List of paths to saved PNG files
            - coordinates (list): List of (x1, y1, x2, y2) tuples indicating
              the position of each crop in the original image coordinates
              
    Example:
        >>> img = np.random.rand(2048, 2048, 3) * 255
        >>> crops, pngs, coords = zoom_img(img, zoom_factor=0.5, target_size=1024, base_filepath='img.czi')
        >>> print(f"Created {len(crops)} overlapping crops")
        >>> print(f"First crop spans: {coords[0]}")
        
    Note:
        - Crops overlap by 50% to avoid missing features at boundaries
        - All crops are resized to target_size × target_size using cubic interpolation
        - Coordinates are in (x1, y1, x2, y2) format relative to original image
        - Works with multi-channel images (BF, RFP, GFP)
        - PNG files are saved with naming: basename_crop_0.png, basename_crop_1.png, etc.
    """
    # Calculate crop dimensions based on zoom factor
    y_size = int(img.shape[0] * zoom_factor)  # Height of each crop
    x_size = int(img.shape[1] * zoom_factor)  # Width of each crop
    
    crops = []        # Store cropped and resized images
    png_paths = []    # Store paths to saved PNG files
    coordinates = []  # Store crop positions in original image
    crop_number = 0   # Counter for crop numbering
    
    # Iterate over image with 50% overlap (stride = size // 2)
    # This ensures we don't miss cellular features at crop boundaries
    for y in range(0, img.shape[0] - y_size + 1, y_size // 2):
        for x in range(0, img.shape[1] - x_size + 1, x_size // 2):
            # Extract sub-region from original image
            sub = img[y:y + y_size, x:x + x_size]
            
            # Resize crop to target size using cubic interpolation
            # Cubic interpolation maintains visual quality better than linear
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
            
            # Add to collection
            crops.append(sub_resized)
            
            # Store crop coordinates in original image space
            # Format: (left, top, right, bottom) for easy mapping back
            coordinates.append((x, y, x + x_size, y + y_size))
            crop_number += 1
    
    return crops, png_paths, coordinates


def predict_and_analyze(model, image_array, image_path, confidence_threshold=0.5, imgsz=1024, crop_id=None):
    """
    Run YOLO prediction and extract statistical features.
    
    Args:
        model: YOLO model
        image_array: Input image as numpy array
        image_path: Path for saving outputs (use .png file directly for CZI)
        confidence_threshold: Minimum confidence for detections
        imgsz: Image size for inference
        
    Returns:
        pandas.DataFrame: Results with statistics
    """
    # Check if image_path is already a PNG file (from CZI conversion)
    if str(image_path).endswith('.png'):
        # Use the existing PNG file directly
        temp_path = str(image_path)
        cleanup_temp = False
    else:
        # Save temporary image for YOLO
        temp_path = str(image_path).replace('.png', '_temp.png').replace('.tif', '_temp.png').replace('.czi', '_temp.png')
        cleanup_temp = True
        
        # Ensure image is in right format for YOLO
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        # Save as BGR for OpenCV compatibility
        if len(image_array.shape) == 3:
            cv2.imwrite(temp_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(temp_path, image_array)
    
    try:
        # Run YOLO prediction
        results = model(temp_path, imgsz=imgsz, visualize=False)
        
        if len(results) == 0 or results[0].boxes is None:
            print(f"No detections found in {image_path}")
            return pd.DataFrame()
        
        # Extract results
        class_names = results[0].names
        nboxes = len(results[0].boxes)
        proba = results[0].boxes.conf.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        masks = results[0].masks
        
        results_dict = {}
        red = clr.Color("red")
        lime = clr.Color("lime")
        colors_list = list(red.range_to(lime, len(class_names)))
        colors = {c: colors_list[i].hex_l for i, c in enumerate(class_names)}
        nclasses = {int(k): k for k in class_names.keys()}
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.imshow(image_array)
        
        rescount = 0
        
        for i in range(nboxes):
            # iterate over the detected objects
            c = int(results[0].boxes.cls[i])
            if proba[i] > confidence_threshold:
                x1, y1, x2, y2 = boxes[i]
                c = int(c)
                
                # Draw bounding box
                plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 
                        color=colors[c], linewidth=2)
                
                # Add label with only two decimal places
                label = str(c) + ' ' + str(proba[i])[:4]
                plt.text(x1, y1, label, color=colors[c])
                
                # Get mask coordinates
                if masks is not None:
                    mask = masks[i].xy[0]
                    mask = mask.astype(int)
                    
                    # Extract pixel values for each channel using mask coordinates
                    bf = image_array[mask[:, 1], mask[:, 0], 0].ravel()
                    rfp = image_array[mask[:, 1], mask[:, 0], 1].ravel()
                    gfp = image_array[mask[:, 1], mask[:, 0], 2].ravel()
                    
                    # Calculate statistics
                    bf_stats = describe(bf)
                    rfp_stats = describe(rfp)
                    gfp_stats = describe(gfp)
                    
                    # Create statistics dictionaries
                    bf_stats_dict = {
                        'bf_mean': bf_stats.mean,
                        'bf_std': bf_stats.variance,
                        'bf_min': bf_stats.minmax[0],
                        'bf_max': bf_stats.minmax[1],
                        'bf_skew': bf_stats.skewness
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
                    
                    # Store results
                    results_dict[rescount] = {
                        'image_path': str(image_path),
                        'crop_id': crop_id if crop_id is not None else 0,
                        'class': c,
                        'proba': proba[i],
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2
                    }
                    results_dict[rescount].update(bf_stats_dict)
                    results_dict[rescount].update(rfp_stats_dict)
                    results_dict[rescount].update(gfp_stats_dict)
                    rescount += 1
        
        plt.title(f'Detections: {rescount}')
        plt.axis('off')
        
        # Save visualization
        output_viz_path = str(image_path).replace('.png', '_annotated.png').replace('.tif', '_annotated.png').replace('.czi', '_annotated.png')
        plt.savefig(output_viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization: {output_viz_path}")
        
        # Convert to DataFrame
        if results_dict:
            df = pd.DataFrame.from_dict(results_dict, orient='index')
            print(df)
            print(len(df), 'objects detected')
            return df
        else:
            return pd.DataFrame()
            
    finally:
        # Clean up temporary file only if we created one
        if cleanup_temp and os.path.exists(temp_path):
            os.remove(temp_path)


def detect_image_format(image_path):
    """Detect image format based on file extension."""
    ext = Path(image_path).suffix.lower()
    if ext in ['.tif', '.tiff']:
        return 'tiff'
    elif ext in ['.czi']:
        return 'czi'
    elif ext in ['.png', '.jpg', '.jpeg']:
        return 'single'
    else:
        raise ValueError(f"Unsupported image format: {ext}")


def find_image_groups(input_dir):
    """
    Find groups of related images (BF, GFP, RFP) in the input directory.
    
    Returns:
        dict: Groups of related images
    """
    # Look for patterns in filenames
    bf_files = glob.glob(os.path.join(input_dir, "*BF*.tif")) + glob.glob(os.path.join(input_dir, "*BF*.TIF"))
    gfp_files = glob.glob(os.path.join(input_dir, "*GFP*.tif")) + glob.glob(os.path.join(input_dir, "*GFP*.TIF"))
    rfp_files = glob.glob(os.path.join(input_dir, "*RFP*.tif")) + glob.glob(os.path.join(input_dir, "*RFP*.TIF"))
    czi_files = glob.glob(os.path.join(input_dir, "*.czi"))
    single_files = glob.glob(os.path.join(input_dir, "*.png")) + glob.glob(os.path.join(input_dir, "*.jpg"))
    
    groups = {}
    
    # Group TIFF files by common identifier
    if bf_files and gfp_files and rfp_files:
        print(f"Found {len(bf_files)} BF, {len(gfp_files)} GFP, {len(rfp_files)} RFP files")
        
        # Extract common identifiers
        for bf_file in bf_files:
            base_name = os.path.basename(bf_file)
            # Try to find matching GFP and RFP files
            identifier = base_name.replace('BF', '').replace('.tif', '').replace('.TIF', '')
            
            matching_gfp = None
            matching_rfp = None
            
            for gfp_file in gfp_files:
                if identifier in os.path.basename(gfp_file):
                    matching_gfp = gfp_file
                    break
            
            for rfp_file in rfp_files:
                if identifier in os.path.basename(rfp_file):
                    matching_rfp = rfp_file
                    break
            
            if matching_gfp and matching_rfp:
                groups[f"tiff_group_{len(groups)}"] = {
                    'type': 'tiff_stack',
                    'bf': bf_file,
                    'gfp': matching_gfp,
                    'rfp': matching_rfp
                }
    
    # Add CZI files
    for czi_file in czi_files:
        groups[f"czi_{len(groups)}"] = {
            'type': 'czi',
            'path': czi_file
        }
    
    # Add single image files
    for single_file in single_files:
        groups[f"single_{len(groups)}"] = {
            'type': 'single',
            'path': single_file
        }
    
    return groups


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(
        description="Annotate images using trained YOLOv8 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process directory with TIFF stacks
  python annotate_images.py --model yolov8n-seg_yfusion.pt --input images/ --output results.csv
  
  # Process with custom confidence threshold
  python annotate_images.py --model yolov8n-seg_yfusion.pt --input images/ --output results.csv --confidence 0.7
  
  # Process CZI images
  python annotate_images.py --model yolov8n-seg_yfusion.pt --input czi_images/ --output results.csv --format czi
  
  # Process with zoomed prediction (overlapping crops)
  python annotate_images.py --model yolov8n-seg_yfusion.pt --input images/ --output results.csv --zoom
  
  # Process with custom zoom factor
  python annotate_images.py --model yolov8n-seg_yfusion.pt --input images/ --output results.csv --zoom --zoom_factor 0.5
  
  # Use configuration file
  python annotate_images.py --config annotate_config.yaml
        """
    )
    
    parser.add_argument('--config', type=str,
                       help='Path to YAML configuration file')
    parser.add_argument('--model', type=str,
                       help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--input', type=str,
                       help='Input directory containing images')
    parser.add_argument('--output', type=str,
                       help='Output CSV file path')
    parser.add_argument('--format', type=str, choices=['auto', 'tiff', 'czi', 'single'], default='auto',
                       help='Image format (auto-detect by default)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--imgsz', type=int, default=1024,
                       help='Image size for inference (default: 1024)')
    parser.add_argument('--crop', type=int, default=1024,
                       help='Crop size for input images (default: 1024)')
    parser.add_argument('--zoom', action='store_true',
                       help='Use zoomed prediction with overlapping crops')
    parser.add_argument('--zoom_factor', type=float, default=40/60,
                       help='Zoom factor for cropping (default: 0.667)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        # Override with command-line arguments if provided
        for key, value in config.items():
            if key == 'zoom' and not args.zoom:
                args.zoom = value
            elif key == 'verbose' and not args.verbose:
                args.verbose = value
            elif not hasattr(args, key) or getattr(args, key) is None or getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)
    
    # Validate required arguments
    required = ['model', 'input', 'output']
    missing = [arg for arg in required if not getattr(args, arg, None)]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}. Provide via --config or command line.")
    
    # Validate inputs
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input directory not found: {args.input}")
    
    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Find image groups
    if args.format == 'auto':
        print("Auto-detecting image format...")
        image_groups = find_image_groups(args.input)
    else:
        # Manual format specification
        image_groups = {}
        if args.format == 'tiff':
            groups = find_image_groups(args.input)
            image_groups = {k: v for k, v in groups.items() if v['type'] == 'tiff_stack'}
        elif args.format == 'czi':
            czi_files = glob.glob(os.path.join(args.input, "*.czi"))
            for i, czi_file in enumerate(czi_files):
                image_groups[f"czi_{i}"] = {'type': 'czi', 'path': czi_file}
        elif args.format == 'single':
            single_files = glob.glob(os.path.join(args.input, "*.png")) + glob.glob(os.path.join(args.input, "*.jpg"))
            for i, single_file in enumerate(single_files):
                image_groups[f"single_{i}"] = {'type': 'single', 'path': single_file}
    
    if not image_groups:
        print("No valid image groups found!")
        return
    
    print(f"Found {len(image_groups)} image groups to process")
    
    all_results = []
    
    # Process each image group
    for group_name, group_info in tqdm.tqdm(image_groups.items(), desc="Processing images"):
        print(f"\nProcessing: {group_name}")
        
        try:
            if group_info['type'] == 'tiff_stack':
                # Process TIFF stack
                stacked_images = process_tiff_stack(
                    group_info['bf'], group_info['gfp'], group_info['rfp'],
                    crop=args.crop, verbose=args.verbose
                )
                
                # Process each frame
                for frame_idx, stacked_img in enumerate(stacked_images):
                    if args.zoom:
                        # Zoomed prediction with multiple crops
                        frame_filepath = f"{group_name}_frame_{frame_idx}"
                        crops, png_paths, coordinates = zoom_img(stacked_img, args.zoom_factor, args.imgsz, base_filepath=frame_filepath)
                        print(f"  Created {len(crops)} zoomed crops with {len(png_paths)} PNG files for frame {frame_idx}")
                        
                        for crop_idx, (crop, png_path, coord) in enumerate(zip(crops, png_paths, coordinates)):
                            crop_name = png_path
                            df = predict_and_analyze(
                                model, crop, crop_name,
                                confidence_threshold=args.confidence,
                                imgsz=args.imgsz,
                                crop_id=crop_idx
                            )
                            
                            if not df.empty:
                                df['group_name'] = group_name
                                df['frame_index'] = frame_idx
                                df['bf_path'] = group_info['bf']
                                df['gfp_path'] = group_info['gfp']
                                df['rfp_path'] = group_info['rfp']
                                df['crop_x1'] = coord[0]
                                df['crop_y1'] = coord[1]
                                df['crop_x2'] = coord[2]
                                df['crop_y2'] = coord[3]
                                all_results.append(df)
                    else:
                        # Normal prediction on full frame
                        frame_name = f"{group_name}_frame_{frame_idx}"
                        df = predict_and_analyze(
                            model, stacked_img, frame_name,
                            confidence_threshold=args.confidence,
                            imgsz=args.imgsz,
                            crop_id=0
                        )
                        
                        if not df.empty:
                            df['group_name'] = group_name
                            df['frame_index'] = frame_idx
                            df['bf_path'] = group_info['bf']
                            df['gfp_path'] = group_info['gfp']
                            df['rfp_path'] = group_info['rfp']
                            all_results.append(df)
            
            elif group_info['type'] == 'czi':
                # Process CZI file
                if not CZI_SUPPORT:
                    print(f"Skipping CZI file {group_info['path']} - CZI support not available")
                    continue
                
                # Load CZI and get PNG path
                img_array, png_path = load_czi_with_imagej(group_info['path'], resize=(args.crop, args.crop))
                if img_array.dtype != np.uint8:
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                
                if args.zoom:
                    # Zoomed prediction with multiple crops
                    crops, png_paths, coordinates = zoom_img(img_array, args.zoom_factor, args.imgsz, base_filepath=group_info['path'])
                    print(f"  Created {len(crops)} zoomed crops with {len(png_paths)} PNG files for {group_name}")
                    
                    for crop_idx, (crop, png_path, coord) in enumerate(zip(crops, png_paths, coordinates)):
                        crop_name = png_path
                        df = predict_and_analyze(
                            model, crop, crop_name,
                            confidence_threshold=args.confidence,
                            imgsz=args.imgsz,
                            crop_id=crop_idx
                        )
                        
                        if not df.empty:
                            df['group_name'] = group_name
                            df['frame_index'] = 0
                            df['source_path'] = group_info['path']
                            df['crop_x1'] = coord[0]
                            df['crop_y1'] = coord[1]
                            df['crop_x2'] = coord[2]
                            df['crop_y2'] = coord[3]
                            all_results.append(df)
                else:
                    # Normal prediction on full image - use PNG path directly
                    df = predict_and_analyze(
                        model, img_array, png_path,
                        confidence_threshold=args.confidence,
                        imgsz=args.imgsz,
                        crop_id=0
                    )
                    
                    if not df.empty:
                        df['group_name'] = group_name
                        df['frame_index'] = 0
                        df['source_path'] = group_info['path']
                        all_results.append(df)
            
            elif group_info['type'] == 'single':
                # Process single image
                img = Image.open(group_info['path'])
                img_array = np.array(img)
                
                # Ensure 3 channels
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.shape[2] == 4:  # RGBA
                    img_array = img_array[:, :, :3]
                
                if args.zoom:
                    # Zoomed prediction with multiple crops
                    crops, png_paths, coordinates = zoom_img(img_array, args.zoom_factor, args.imgsz, base_filepath=group_info['path'])
                    print(f"  Created {len(crops)} zoomed crops with {len(png_paths)} PNG files for {group_name}")
                    
                    for crop_idx, (crop, png_path, coord) in enumerate(zip(crops, png_paths, coordinates)):
                        crop_name = png_path
                        df = predict_and_analyze(
                            model, crop, crop_name,
                            confidence_threshold=args.confidence,
                            imgsz=args.imgsz,
                            crop_id=crop_idx
                        )
                        
                        if not df.empty:
                            df['group_name'] = group_name
                            df['frame_index'] = 0
                            df['source_path'] = group_info['path']
                            df['crop_x1'] = coord[0]
                            df['crop_y1'] = coord[1]
                            df['crop_x2'] = coord[2]
                            df['crop_y2'] = coord[3]
                            all_results.append(df)
                else:
                    # Normal prediction on full image
                    df = predict_and_analyze(
                        model, img_array, group_info['path'],
                        confidence_threshold=args.confidence,
                        imgsz=args.imgsz,
                        crop_id=0
                    )
                    
                    if not df.empty:
                        df['group_name'] = group_name
                        df['frame_index'] = 0
                        df['source_path'] = group_info['path']
                        all_results.append(df)
        
        except Exception as e:
            print(f"Error processing {group_name}: {str(e)}")
            continue
    
    # Combine all results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        # Save to CSV
        final_df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
        print(f"Total detections: {len(final_df)}")
        
        # Print summary statistics
        if 'class' in final_df.columns:
            print("\nDetection summary by class:")
            print(final_df['class'].value_counts().sort_index())
            
        if 'confidence' in final_df.columns:
            print(f"\nConfidence statistics:")
            print(f"Mean: {final_df['confidence'].mean():.3f}")
            print(f"Min:  {final_df['confidence'].min():.3f}")
            print(f"Max:  {final_df['confidence'].max():.3f}")
    else:
        print("No detections found in any images!")


if __name__ == "__main__":
    main()