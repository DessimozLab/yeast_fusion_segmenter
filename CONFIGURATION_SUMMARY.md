# Configuration Files - Summary

## What Was Added

### 1. Configuration File Support in Scripts

Both prediction scripts now support YAML configuration files:

#### Modified Files:
- **batch_predict.py**: Added `--config` argument and config loading logic
- **annotate_images.py**: Added `--config` argument and config loading logic

#### New Capabilities:
- Load all parameters from YAML files
- Override config values with command-line arguments
- Mix config files and CLI for flexibility
- Better parameter management for complex workflows

### 2. Configuration Files

Two example configuration files using `yolov8s-seg_yfusion.pt`:

#### batch_predict_config.yaml
- Configured for batch processing PNG images
- Input: `datasets/test/images`
- Output: `batch_predictions.csv`
- Crop size: 1024
- Zoom disabled by default

#### annotate_images_config.yaml
- Configured for annotating images with auto-detection
- Input: `images_CNN_clean/`
- Output: `annotation_results.csv`
- Confidence threshold: 0.5
- Supports TIFF stacks, CZI files, and single images

### 3. Test Suite

#### New Test File: tests/test_config_files.py
- Validates config file existence
- Checks YAML structure
- Verifies parameter ranges
- Confirms model specification
- Tests config loading/parsing

#### Test Results:
```
Ran 6 tests in 0.019s
OK - All tests passed
```

### 4. Documentation

#### CONFIG_GUIDE.md
Comprehensive guide covering:
- Basic usage with config files
- Parameter override examples
- Custom configuration creation
- Best practices
- Troubleshooting
- Advanced usage patterns
- Migration guide from CLI-only

#### Updated tests/README.md
- Added test_config_files.py documentation
- Added configuration file usage section
- Updated contributing guidelines

## Usage Examples

### Using batch_predict.py with config:
```bash
# Basic usage
python batch_predict.py --config batch_predict_config.yaml

# Override parameters
python batch_predict.py --config batch_predict_config.yaml --crop 512 --zoom
```

### Using annotate_images.py with config:
```bash
# Basic usage
python annotate_images.py --config annotate_images_config.yaml

# Override confidence and enable verbose
python annotate_images.py --config annotate_images_config.yaml --confidence 0.7 --verbose
```

## Benefits

1. **Reproducibility**: Save exact parameters used for each run
2. **Ease of Use**: Shorter command lines for complex setups
3. **Version Control**: Track parameter changes over time
4. **Collaboration**: Share configs with team members
5. **Flexibility**: Still supports full CLI usage
6. **Documentation**: Self-documenting with inline comments

## File Structure

```
yeast_fusion_segmenter/
├── batch_predict.py                 # Modified: Added config support
├── annotate_images.py               # Modified: Added config support
├── batch_predict_config.yaml        # New: Example config
├── annotate_images_config.yaml      # New: Example config
├── CONFIG_GUIDE.md                  # New: Comprehensive guide
└── tests/
    ├── test_config_files.py         # New: Config validation tests
    ├── test_prediction_outputs.py   # Existing: Output format tests
    ├── test_integration.py          # Existing: Integration tests
    └── README.md                    # Updated: Added config info
```

## Model Specification

Both config files are set to use **yolov8s-seg_yfusion.pt**, which is:
- A YOLOv8 segmentation model (small variant)
- Trained on yeast fusion data
- Suitable for production use
- Balanced between speed and accuracy

## Testing

All configuration features are fully tested:
```bash
# Run config tests
python tests/test_config_files.py

# Run all tests
python tests/run_tests.py
```

## Next Steps

1. **Customize configs** for your specific data paths
2. **Create project-specific configs** for different experiments
3. **Version control configs** alongside your data processing scripts
4. **Share configs** with collaborators for reproducible results

## Support

- See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for detailed usage
- See [tests/README.md](tests/README.md) for testing information
- Check inline comments in config files for parameter explanations
