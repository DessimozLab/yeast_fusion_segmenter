#!/usr/bin/env python3

import os
import glob
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence, ImageEnhance
import cv2
import shutil
import random
import tqdm
import copy
import skimage.measure as measure
import logging
from pathlib import Path
import pickle
import warnings
import sys
import re

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_preparation.log')
    ]
)
logger = logging.getLogger('DataPreparation')

# HDF5 annotations follow the seven 1000-wide phenotype bins defined in
# ``segment_retrain(1).ipynb``. Their order is a model contract.
CLASS_NAMES = (
    'f',      # free cells
    'h',      # hyphae
    'lmcf',   # lysis: medium, cytoplasmic fluorescence
    'lmsgfp', # lysis: medium, strong GFP
    'lsgfp',  # lysis: strong GFP
    'dip',    # diploid
    'd2',     # doublet / d2 phenotype
)
CLASS_BIN_WIDTH = 1000
_FRAME_SUFFIX_RE = re.compile(r'__f(?P<index>\d{4})$')

def parse_args():
    """Parse command line arguments for data preparation"""
    parser = argparse.ArgumentParser(description='Prepare image data for training YOLO model')
    
    # Input/Output arguments
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing the source images')
    parser.add_argument('--output-dir', type=str, default='datasets',
                        help='Output directory for processed datasets')
    
    # Processing options
    parser.add_argument('--file-format', type=str, choices=['tiff', 'czi', 'raw'], required=True,
                        help='Input format; use raw for the canonical data/raw layout')
    parser.add_argument('--magnification', choices=['all', '40x'], default='all',
                        help='For --file-format raw, prepare all images or only 40x')
    parser.add_argument('--source-format', choices=['all', 'czi', 'tiff'], default='all',
                        help='For --file-format raw, restrict the selected source image format')
    parser.add_argument('--dataset-info', type=str, default=None,
                        help='Where to save the reproducible dataset-info JSON (raw input only)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume a partially prepared canonical raw dataset')
    parser.add_argument('--crop-size', type=int, default=1024,
                        help='Size to crop images (default: 1024)')
    parser.add_argument('--aug-count', type=int, default=5,
                        help='Number of augmented images to generate per input image')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Fraction of images to use for validation')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Fraction of images to use for testing')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--brightness', type=float, default=1.0,
                        help='Brightness adjustment factor')
    parser.add_argument('--contrast', type=float, default=1.0,
                        help='Contrast adjustment factor')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output with visualizations')
    
    return parser.parse_args()

def create_directory_structure(output_dir, resume=False):
    """Create the directory structure for the dataset"""
    logger.info(f"Creating directory structure in {output_dir}")
    
    # Remove existing directory if it exists
    if os.path.exists(output_dir):
        if resume:
            logger.info("Resuming existing output directory %s", output_dir)
            return
        logger.warning(f"Output directory {output_dir} already exists. Removing it.")
        shutil.rmtree(output_dir)
    
    # Create main directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create train/val/test directories with images and labels subdirectories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    logger.info("Directory structure created successfully")


def validate_split_arguments(args):
    """Validate the explicit train/validation/test split requested by the CLI."""
    if not 0 <= args.val_split < 1 or not 0 <= args.test_split < 1:
        raise ValueError("--val-split and --test-split must be in [0, 1)")
    if args.val_split + args.test_split >= 1:
        raise ValueError("--val-split plus --test-split must be less than 1")

def yield_frames(img, crop=1024, verbose=False, scaler=True):
    """Extract and normalize frames from an image"""
    for i, page in enumerate(ImageSequence.Iterator(img)):
        if verbose:
            plt.figure(figsize=(8, 8))
            plt.imshow(np.array(page))
            plt.title(f"Frame {i}")
            plt.show()
        
        page_array = np.array(page)
        if crop is not None:
            page_array = page_array[0:crop, 0:crop]
        
        if scaler:
            # Normalize to 0-255 range
            if page_array.max() > page_array.min():
                page_array = (page_array - page_array.min()) / (page_array.max() - page_array.min()) * 255
            else:
                page_array = np.zeros_like(page_array)
        
        yield page_array.astype(np.uint8)

def adjust_brightness_contrast(image, brightness=1.0, contrast=1.0):
    """Adjust brightness and contrast of an image"""
    # Convert numpy array to PIL Image if necessary
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    
    # Adjust contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    
    return image


def center_crop_or_pad(image, size=1024):
    """Notebook-compatible center crop with zero padding for smaller inputs."""
    height, width = image.shape[:2]
    y0 = max((height - size) // 2, 0)
    x0 = max((width - size) // 2, 0)
    cropped = image[y0:min(y0 + size, height), x0:min(x0 + size, width), ...]
    output_shape = (size, size) if image.ndim == 2 else (size, size, image.shape[2])
    output = np.zeros(output_shape, dtype=image.dtype)
    if image.ndim == 2:
        output[:cropped.shape[0], :cropped.shape[1]] = cropped
    else:
        output[:cropped.shape[0], :cropped.shape[1], :] = cropped
    return output

def split_mask(mask, crop=1024, num_classes=len(CLASS_NAMES)):
    """Split an encoded mask with the notebook's seven-bin logic.

    Class ``i`` is exactly ``i * 1000 <= value < (i + 1) * 1000``. Zeros in
    the class-0 working mask are background and are ignored by contour export.
    """
    if num_classes != len(CLASS_NAMES):
        raise ValueError(f"Expected {len(CLASS_NAMES)} phenotype classes, got {num_classes}")
    mask = mask[0:crop, 0:crop]
    masks = []
    for class_id in range(num_classes):
        class_mask = copy.deepcopy(mask)
        lower = class_id * CLASS_BIN_WIDTH
        upper = (class_id + 1) * CLASS_BIN_WIDTH
        class_mask[(class_mask < lower) | (class_mask >= upper)] = 0
        masks.append(class_mask)
    return masks

def output_contours(m, cl, verbose=False):
    """Extract contours from a mask and format them for YOLO training"""
    c = []
    for val in list(np.unique(m)):
        if val == 0:  # Skip background
            continue
        sub = copy.deepcopy(m)
        sub[sub != val] = 0
        c += measure.find_contours(sub, 0.9)
    
    contours = c
    
    if verbose:
        plt.figure(figsize=(8, 8))
        plt.imshow(m)
        plt.title(f'Contours Class {cl}')
        for n, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
        plt.show()

    # Output contours to YOLO format
    # <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
    lines = []
    
    for c in contours:
        if len(c) < 3:  # Skip contours with too few points
            continue
            
        coords = []
        for i in range(0, c.shape[0]):
            coords.append((float(c[i][1]) / m.shape[0]))
            coords.append((float(c[i][0]) / m.shape[1]))
        
        line = f"{cl} " + " ".join([str(coord) for coord in coords]) + "\n"
        lines.append(line)
    
    return lines

def mask_to_contour_file(mask, output_file, verbose=False):
    """Convert every notebook phenotype bin in an HDF5 mask to YOLO polygons."""
    masks = mask if isinstance(mask, (list, tuple)) else split_mask(mask)
    if len(masks) != len(CLASS_NAMES):
        raise ValueError(f"Expected {len(CLASS_NAMES)} class masks, got {len(masks)}")
    lines = []
    for class_id, class_mask in enumerate(masks):
        lines.extend(output_contours(class_mask, class_id, verbose=verbose))
    
    with open(output_file, 'w') as f:
        for l in lines:
            f.write(l)
    
    return output_file


def mask_boundary_alignment_score(mask, rgb, edge_channel=1):
    """Measure mask-boundary agreement with one fluorescence image channel.

    This reproduces the CZI orientation diagnostic in
    ``segment_retrain(1).ipynb``.
    """
    mask_binary = (mask > 0).astype(np.uint8) * 255
    contour_edges = cv2.Canny(mask_binary, 50, 150) > 0
    if not contour_edges.any():
        return 0.0, float("inf")
    channel = cv2.GaussianBlur(rgb[:, :, edge_channel], (5, 5), 0)
    median = np.median(channel)
    edge_map = cv2.Canny(channel, int(max(0, 0.66 * median)), int(min(255, 1.33 * median))) > 0
    dilated_edges = cv2.dilate(edge_map.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    overlap = float((contour_edges & dilated_edges).sum()) / float(contour_edges.sum())
    distance = cv2.distanceTransform((~edge_map).astype(np.uint8), cv2.DIST_L2, 3)
    return overlap, float(distance[contour_edges].mean())


def align_czi_mask_orientation(mask, rgb, improvement_threshold=0.10):
    """Apply the notebook's per-CZI flip selection when it clearly improves alignment."""
    candidates = {
        "orig": mask,
        "flip_ud": np.flipud(mask),
        "flip_lr": np.fliplr(mask),
        "flip_udlr": np.fliplr(np.flipud(mask)),
    }
    scored = {
        name: (*mask_boundary_alignment_score(candidate, rgb), candidate)
        for name, candidate in candidates.items()
    }
    original_overlap, original_distance, _ = scored["orig"]
    selected, (best_overlap, best_distance, best_mask) = sorted(
        scored.items(), key=lambda item: (-item[1][0], item[1][1])
    )[0]
    should_apply = (
        selected != "orig"
        and best_overlap > original_overlap + improvement_threshold
        and best_distance < original_distance
    )
    return (best_mask if should_apply else mask), (selected if should_apply else "orig"), {
        name: (overlap, distance) for name, (overlap, distance, _) in scored.items()
    }


def load_annotation_mask(annotation_path, materialized_sample_id):
    """Load the HDF5 frame corresponding to a materialized PNG record.

    TIFF stacks become ``<sample>__f0000.png``, ``__f0001.png``, and so on.
    Their HDF5 datasets use the same frame order, so each PNG must consume the
    matching mask rather than the first non-empty mask in the file. A CZI or
    single-frame TIFF has no suffix and uses frame zero.
    """
    match = _FRAME_SUFFIX_RE.search(materialized_sample_id)
    frame_index = int(match['index']) if match else 0
    with h5py.File(annotation_path, 'r') as mask_file:
        frames = [
            np.asarray(mask_file[group][frame], dtype=np.uint16)
            for group in mask_file.keys()
            for frame in mask_file[group]
        ]
    if frame_index >= len(frames):
        raise ValueError(
            f"Annotation {annotation_path} has {len(frames)} frames but "
            f"{materialized_sample_id} requires frame {frame_index}"
        )
    return frames[frame_index]

def random_rotation(image, masks, angle_range):
    """Apply random rotation to image and masks"""
    angle = random.uniform(-angle_range, angle_range)
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Rotate image
    image = image.rotate(angle)
    
    # Convert each mask to PIL Image and rotate
    masks = [Image.fromarray(m) for m in masks]
    masks = [m.rotate(angle) for m in masks]
    
    return np.array(image), masks

def random_flip(image, masks):
    """Randomly flip image and masks horizontally"""
    if random.random() > 0.5:
        # Flip image
        if isinstance(image, np.ndarray):
            image = cv2.flip(image, 1)
        else:
            image = np.array(image)
            image = cv2.flip(image, 1)
            image = Image.fromarray(image)
        
        # Flip masks
        masks = [cv2.flip(np.array(m), 1) for m in masks]
    
    return image, masks

def augment_and_resize(image, masks, angle_range=180, crop_size=900, size=(1024, 1024)):
    """Apply augmentation to image and masks"""
    # Ensure image is a numpy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Apply random rotation
    image, masks = random_rotation(image, masks, angle_range)
    
    # Apply random flip
    image, masks = random_flip(image, masks)
    
    # Convert masks to numpy arrays
    masks = [np.array(m) for m in masks]
    
    return image, masks

def process_tiff_files(args):
    """Process TIFF files and prepare for YOLO training"""
    logger.info("Processing TIFF files")
    
    # Find all TIFF files in input directory
    input_dir = Path(args.input_dir)
    
    # Look for brightfield, GFP, and RFP images with standard naming patterns
    bf_images = sorted(list(input_dir.glob('*[Bb][Ff]*.tif')) + list(input_dir.glob('*[Bb][Ff]*.TIF')))
    gfp_images = sorted(list(input_dir.glob('*[Gg][Ff][Pp]*.tif')) + list(input_dir.glob('*[Gg][Ff][Pp]*.TIF')))
    rfp_images = sorted(list(input_dir.glob('*[Rr][Ff][Pp]*.tif')) + list(input_dir.glob('*[Rr][Ff][Pp]*.TIF')))
    mask_files = sorted(list(input_dir.glob('*[Mm]ask*.h5')))
    
    logger.info(f"Found {len(bf_images)} brightfield images")
    logger.info(f"Found {len(gfp_images)} GFP images")
    logger.info(f"Found {len(rfp_images)} RFP images")
    logger.info(f"Found {len(mask_files)} mask files")
    
    if len(bf_images) == 0 or len(mask_files) == 0:
        logger.error("No images or mask files found. Please check the input directory.")
        return
    
    # Create a dataset dictionary
    dataset = {}
    for i, bf_path in enumerate(bf_images):
        # Find matching mask file
        mask_path = None
        for mask_file in mask_files:
            if str(bf_path.stem).split('_')[0] in str(mask_file):
                mask_path = mask_file
                break
        
        if mask_path is None:
            logger.warning(f"No matching mask found for {bf_path}. Skipping.")
            continue
        
        # Find matching fluorescence images if available
        gfp_path = None
        for gfp_file in gfp_images:
            if str(bf_path.stem).split('_')[0] in str(gfp_file):
                gfp_path = gfp_file
                break
        
        rfp_path = None
        for rfp_file in rfp_images:
            if str(bf_path.stem).split('_')[0] in str(rfp_file):
                rfp_path = rfp_file
                break
        
        if gfp_path is None or rfp_path is None:
            logger.warning(f"No matching fluorescence images found for {bf_path}. Skipping.")
            continue
        
        dataset[i] = {
            'img': str(bf_path),
            'mask': str(mask_path),
            'gfp': str(gfp_path),
            'rfp': str(rfp_path)
        }
    
    logger.info(f"Created dataset with {len(dataset)} valid entries")
    
    # Process each dataset entry
    image_count = 0
    for idx, entry in tqdm.tqdm(dataset.items(), desc="Processing images"):
        # Load mask file
        try:
            mask_file = h5py.File(entry['mask'], 'r')
            mask_found = False
            
            for group in mask_file.keys():
                for frame in mask_file[group]:
                    mask = np.array(mask_file[group][frame], dtype=np.uint16)
                    
                    if np.sum(mask) > 0:
                        mask = mask[0:args.crop_size, 0:args.crop_size]
                        mask_found = True
                        break
                
                if mask_found:
                    break
            
            if not mask_found:
                logger.warning(f"No valid mask data found in {entry['mask']}. Skipping.")
                continue
            
            # Load images
            bf_img = Image.open(entry['img'])
            bf_frames = list(yield_frames(bf_img, crop=args.crop_size, verbose=args.verbose))
            
            gfp_img = Image.open(entry['gfp'])
            gfp_frames = list(yield_frames(gfp_img, crop=args.crop_size, verbose=args.verbose))
            
            rfp_img = Image.open(entry['rfp'])
            rfp_frames = list(yield_frames(rfp_img, crop=args.crop_size, verbose=args.verbose))
            
            # Process each frame
            for frame_idx in range(min(len(bf_frames), len(gfp_frames), len(rfp_frames))):
                # Stack channels
                stacked_image = np.stack([
                    bf_frames[frame_idx],
                    gfp_frames[frame_idx],
                    rfp_frames[frame_idx]
                ], axis=-1)
                
                # Save original image
                output_image_path = os.path.join(args.output_dir, 'train', 'images', f'img_{image_count:06d}.png')
                cv2.imwrite(output_image_path, stacked_image)
                
                # Save original mask to contour file
                masks = split_mask(mask)
                output_label_path = os.path.join(args.output_dir, 'train', 'labels', f'img_{image_count:06d}.txt')
                mask_to_contour_file(masks, output_label_path, verbose=args.verbose)
                
                image_count += 1
                
                # Generate augmentations
                for aug_idx in range(args.aug_count):
                    aug_image, aug_masks = augment_and_resize(
                        stacked_image, masks,
                        angle_range=180,
                        crop_size=args.crop_size
                    )
                    
                    # Save augmented image
                    aug_output_path = os.path.join(args.output_dir, 'train', 'images', f'img_{image_count:06d}.png')
                    cv2.imwrite(aug_output_path, aug_image)
                    
                    # Save augmented mask to contour file
                    aug_label_path = os.path.join(args.output_dir, 'train', 'labels', f'img_{image_count:06d}.txt')
                    mask_to_contour_file(aug_masks, aug_label_path, verbose=args.verbose)
                    
                    image_count += 1
            
            mask_file.close()
            
        except Exception as e:
            logger.error(f"Error processing {entry['img']}: {str(e)}")
    
    logger.info(f"Generated {image_count} total images for training")
    return image_count

def process_czi_files(args):
    """Process CZI files and prepare for YOLO training"""
    logger.info("Processing CZI files")
    
    # Check if required packages are installed
    try:
        import czifile
    except ImportError:
        logger.error("czifile package is required for CZI processing. Please install it using pip install czifile.")
        return 0
    
    try:
        import imagej
        ij = imagej.init('sc.fiji:fiji', headless=True)
        use_imagej = True
        logger.info("Using ImageJ for CZI processing")
    except ImportError:
        logger.warning("ImageJ is not available. Falling back to czifile for CZI processing.")
        use_imagej = False
    
    # Find all CZI files in input directory
    input_dir = Path(args.input_dir)
    czi_files = list(input_dir.glob('*.czi'))
    
    logger.info(f"Found {len(czi_files)} CZI files")
    
    if len(czi_files) == 0:
        logger.error("No CZI files found. Please check the input directory.")
        return 0
    
    # Process each CZI file
    image_count = 0
    for czi_file in tqdm.tqdm(czi_files, desc="Processing CZI files"):
        try:
            if use_imagej:
                # Open using ImageJ
                dataset = ij.io().open(str(czi_file))
                img_data = ij.py.from_java(dataset)
                
                # Handle different dimensionality
                if len(img_data.shape) == 5:  # T, C, Z, Y, X
                    logger.info(f"CZI shape: {img_data.shape}")
                    
                    # Extract channels (assuming first timepoint, first Z-slice)
                    bf_channel = np.array(img_data[0, 0, 0])  # T=0, C=0, Z=0
                    
                    # Check if we have enough channels for fluorescence
                    if img_data.shape[1] >= 3:
                        gfp_channel = np.array(img_data[0, 1, 0])  # T=0, C=1, Z=0
                        rfp_channel = np.array(img_data[0, 2, 0])  # T=0, C=2, Z=0
                    else:
                        # If not enough channels, duplicate the BF channel
                        logger.warning(f"Not enough channels in {czi_file}. Using BF for all channels.")
                        gfp_channel = bf_channel.copy()
                        rfp_channel = bf_channel.copy()
                
                elif len(img_data.shape) == 4:  # C, Z, Y, X
                    logger.info(f"CZI shape: {img_data.shape}")
                    
                    bf_channel = np.array(img_data[0, 0])  # C=0, Z=0
                    if img_data.shape[0] >= 3:
                        gfp_channel = np.array(img_data[1, 0])  # C=1, Z=0
                        rfp_channel = np.array(img_data[2, 0])  # C=2, Z=0
                    else:
                        logger.warning(f"Not enough channels in {czi_file}. Using BF for all channels.")
                        gfp_channel = bf_channel.copy()
                        rfp_channel = bf_channel.copy()
                
                else:
                    logger.error(f"Unexpected dimensionality in {czi_file}: {img_data.shape}")
                    continue
            else:
                # Open using czifile
                img = czifile.imread(str(czi_file))
                
                # Handle different dimensionality
                if len(img.shape) == 5:  # T, C, Z, Y, X
                    bf_channel = np.squeeze(img[0, 0, 0])
                    
                    if img.shape[1] >= 3:
                        gfp_channel = np.squeeze(img[0, 1, 0])
                        rfp_channel = np.squeeze(img[0, 2, 0])
                    else:
                        gfp_channel = bf_channel.copy()
                        rfp_channel = bf_channel.copy()
                
                elif len(img.shape) == 4:  # C, Z, Y, X
                    bf_channel = np.squeeze(img[0, 0])
                    
                    if img.shape[0] >= 3:
                        gfp_channel = np.squeeze(img[1, 0])
                        rfp_channel = np.squeeze(img[2, 0])
                    else:
                        gfp_channel = bf_channel.copy()
                        rfp_channel = bf_channel.copy()
                
                else:
                    logger.error(f"Unexpected dimensionality in {czi_file}: {img.shape}")
                    continue
            
            # Normalize each channel to 0-255
            bf_normalized = ((bf_channel - bf_channel.min()) / 
                          (bf_channel.max() - bf_channel.min() + 1e-10) * 255).astype(np.uint8)
            gfp_normalized = ((gfp_channel - gfp_channel.min()) / 
                           (gfp_channel.max() - gfp_channel.min() + 1e-10) * 255).astype(np.uint8)
            rfp_normalized = ((rfp_channel - rfp_channel.min()) / 
                          (rfp_channel.max() - rfp_channel.min() + 1e-10) * 255).astype(np.uint8)
            
            # Crop to specified size
            h, w = bf_normalized.shape
            crop_size = min(args.crop_size, h, w)
            
            bf_cropped = bf_normalized[:crop_size, :crop_size]
            gfp_cropped = gfp_normalized[:crop_size, :crop_size]
            rfp_cropped = rfp_normalized[:crop_size, :crop_size]
            
            # Stack channels
            stacked_image = np.stack([bf_cropped, gfp_cropped, rfp_cropped], axis=-1)
            
            # Apply brightness/contrast adjustment if specified
            if args.brightness != 1.0 or args.contrast != 1.0:
                stacked_image = np.array(adjust_brightness_contrast(
                    Image.fromarray(stacked_image),
                    brightness=args.brightness,
                    contrast=args.contrast
                ))
            
            # Since we don't have masks for CZI files, we'll just save them to the test folder
            # They'll be used for inference but not training
            output_image_path = os.path.join(args.output_dir, 'test', 'images', f'img_{image_count:06d}.png')
            cv2.imwrite(output_image_path, stacked_image)
            
            # Generate an empty label file
            open(os.path.join(args.output_dir, 'test', 'labels', f'img_{image_count:06d}.txt'), 'w').close()
            
            image_count += 1
            
            # Show results if verbose
            if args.verbose:
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 4, 1)
                plt.imshow(bf_cropped, cmap='gray')
                plt.title('Brightfield')
                plt.axis('off')
                
                plt.subplot(1, 4, 2)
                plt.imshow(gfp_cropped, cmap='Greens')
                plt.title('GFP')
                plt.axis('off')
                
                plt.subplot(1, 4, 3)
                plt.imshow(rfp_cropped, cmap='Reds')
                plt.title('RFP')
                plt.axis('off')
                
                plt.subplot(1, 4, 4)
                plt.imshow(stacked_image)
                plt.title('Combined')
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()
        
        except Exception as e:
            logger.error(f"Error processing {czi_file}: {str(e)}")
    
    logger.info(f"Processed {image_count} CZI files")
    return image_count


def process_canonical_raw_files(args):
    """Create a YOLO dataset from canonical raw data via MicroscopyImageDataset.

    This is the supported training entry point for ``data/raw``.  Source files
    are converted to PNG before they are copied into a split, and labels are
    resolved from the annotation path stored in the record metadata.
    """
    from image_dataset import MicroscopyImageDataset

    magnifications = ('40x',) if args.magnification == '40x' else ('40x', 'other')
    source_formats = ('czi', 'tiff') if args.source_format == 'all' else (args.source_format,)
    raw_dataset = MicroscopyImageDataset(
        args.input_dir, magnifications=magnifications, source_formats=source_formats
    )
    png_records = raw_dataset.materialize_pngs()
    image_count = 0
    for record in tqdm.tqdm(png_records, desc="Preparing canonical raw images"):
        split = 'train' if record.annotation_path else 'test'
        output_base = f"{record.magnification}_{record.sample_id}"
        output_image = os.path.join(args.output_dir, split, 'images', output_base + '.png')
        output_label = os.path.join(args.output_dir, split, 'labels', output_base + '.txt')
        # Match the notebook exactly: center crop/pad RGB and its HDF5 mask
        # together before contour extraction. Rewriting is intentional so a
        # resumed legacy output cannot retain full-frame images or collapsed
        # three-class labels.
        with Image.open(record.sources['png']) as source_image:
            rgb = np.asarray(source_image.convert('RGB'))
        rgb = center_crop_or_pad(rgb, size=args.crop_size)
        Image.fromarray(rgb).save(output_image)
        if record.annotation_path:
            mask = load_annotation_mask(record.annotation_path, record.sample_id)
            if not mask.any():
                logger.warning("Empty annotation frame for %s", record.annotation_path)
                open(output_label, 'w').close()
            else:
                mask = center_crop_or_pad(mask, size=args.crop_size)
                if record.source_format == 'czi':
                    mask, orientation, scores = align_czi_mask_orientation(mask, rgb)
                    logger.info(
                        "CZI mask orientation for %s: %s (orig overlap %.3f; selected overlap %.3f)",
                        record.sample_id, orientation, scores['orig'][0], scores[orientation][0],
                    )
                mask_to_contour_file(split_mask(mask, crop=args.crop_size), output_label, verbose=args.verbose)
        else:
            open(output_label, 'w').close()
        image_count += 1
    logger.info("Prepared %s PNGs from canonical raw data", image_count)
    return image_count, raw_dataset

def create_dataset_yaml(output_dir):
    """Create a YAML file with dataset configuration for YOLO training"""
    class_lines = '\n'.join(f"  {class_id}: {name}" for class_id, name in enumerate(CLASS_NAMES))
    yaml_content = f"""
# YOLOv8 dataset configuration
path: {os.path.abspath(output_dir)}
train: train/images
val: val/images
test: test/images

names:
{class_lines}
"""

    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"Created dataset configuration at {yaml_path}")
    return yaml_path

def split_dataset(output_dir, val_split=0.1, test_split=0.1):
    """Split the labelled portion into train/validation/test sets.

    Raw microscopy collections may include many images without a non-empty
    annotation.  Canonical preparation initially puts those images in test so
    they remain available for inference, but they must not consume the tiny
    supervised validation/test budgets.  Therefore hold-outs are selected
    only from images with at least one YOLO annotation; empty-label images
    stay in train (or, for source images without an HDF5 file, in test).
    """
    logger.info("Splitting dataset into train/val/test sets")
    
    # Get all image files in the train folder
    train_image_dir = os.path.join(output_dir, 'train', 'images')
    image_files = [f for f in os.listdir(train_image_dir) if f.endswith('.png')]
    label_dir = os.path.join(output_dir, 'train', 'labels')
    annotated_files = [
        filename for filename in image_files
        if os.path.exists(os.path.join(label_dir, f"{os.path.splitext(filename)[0]}.txt"))
        and os.path.getsize(os.path.join(label_dir, f"{os.path.splitext(filename)[0]}.txt")) > 0
    ]
    
    # Shuffle the images
    random.shuffle(annotated_files)
    
    # Calculate split points
    total = len(annotated_files)
    # A fractional split should not silently disappear for a small microscopy
    # collection. Reserve one image for each requested hold-out split whenever
    # at least one training image can still remain.
    val_count = int(total * val_split)
    test_count = int(total * test_split)
    if val_split > 0 and total >= 3:
        val_count = max(1, val_count)
    if test_split > 0 and total - val_count >= 2:
        test_count = max(1, test_count)
    if val_count + test_count >= total:
        overflow = val_count + test_count - (total - 1)
        if test_count >= overflow:
            test_count -= overflow
        else:
            val_count -= overflow - test_count
            test_count = 0
    
    # Split the files
    val_files = annotated_files[:val_count]
    test_files = annotated_files[val_count:val_count + test_count]
    
    # Move validation files
    for f in val_files:
        base_name = os.path.splitext(f)[0]
        
        # Move image
        src_img = os.path.join(output_dir, 'train', 'images', f)
        dst_img = os.path.join(output_dir, 'val', 'images', f)
        shutil.move(src_img, dst_img)
        
        # Move label
        src_lbl = os.path.join(output_dir, 'train', 'labels', f'{base_name}.txt')
        dst_lbl = os.path.join(output_dir, 'val', 'labels', f'{base_name}.txt')
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)
    
    # Move test files
    for f in test_files:
        base_name = os.path.splitext(f)[0]
        
        # Move image
        src_img = os.path.join(output_dir, 'train', 'images', f)
        dst_img = os.path.join(output_dir, 'test', 'images', f)
        shutil.move(src_img, dst_img)
        
        # Move label
        src_lbl = os.path.join(output_dir, 'train', 'labels', f'{base_name}.txt')
        dst_lbl = os.path.join(output_dir, 'test', 'labels', f'{base_name}.txt')
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)
    
    # Count files in each set
    train_count = len(os.listdir(os.path.join(output_dir, 'train', 'images')))
    val_count = len(os.listdir(os.path.join(output_dir, 'val', 'images')))
    test_count = len(os.listdir(os.path.join(output_dir, 'test', 'images')))
    
    logger.info(f"Dataset split: {train_count} train, {val_count} validation, {test_count} test images")

def main():
    """Main function to prepare data for YOLO training"""
    args = parse_args()
    validate_split_arguments(args)
    
    # Set random seed for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Create directory structure
    create_directory_structure(args.output_dir, resume=args.resume)
    
    # Process files based on format
    if args.file_format == 'tiff':
        process_tiff_files(args)
    elif args.file_format == 'czi':
        process_czi_files(args)
    elif args.file_format == 'raw':
        _, raw_dataset = process_canonical_raw_files(args)
    
    # Split the dataset
    split_dataset(args.output_dir, args.val_split, args.test_split)
    
    # Create dataset YAML
    yaml_path = create_dataset_yaml(args.output_dir)
    if args.file_format == 'raw':
        info_path = args.dataset_info or os.path.join(args.output_dir, 'dataset-info.json')
        raw_dataset.save_dataset_info(
            info_path,
            name=Path(args.output_dir).name,
            yolo_data=yaml_path,
            split={
                "train": round(1 - args.val_split - args.test_split, 10),
                "val": args.val_split,
                "test": args.test_split,
                "random_seed": args.random_seed,
                "unannotated_to_test": True,
            },
            annotation_classes={class_id: name for class_id, name in enumerate(CLASS_NAMES)},
        )
        logger.info("Saved dataset info at %s", info_path)
    
    logger.info("Data preparation completed successfully")
    logger.info(f"Dataset ready for training at: {args.output_dir}")
    logger.info(f"Use the following command to train a model:")
    logger.info(f"  python train_yolo.py --data {yaml_path} --epochs 100 --batch-size 8 --output yolov8n-seg_custom.pt")

if __name__ == "__main__":
    main()
