# Quick Reference - Configuration Files

## TL;DR

```bash
# Use config file
python batch_predict.py --config batch_predict_config.yaml
python annotate_images.py --config annotate_images_config.yaml

# Override from command line
python batch_predict.py --config batch_predict_config.yaml --crop 512
python annotate_images.py --config annotate_images_config.yaml --confidence 0.7
```

## Config File Locations

- `batch_predict_config.yaml` - Batch prediction config
- `annotate_images_config.yaml` - Image annotation config

## Key Parameters

### batch_predict_config.yaml
```yaml
model: yolov8s-seg_yfusion.pt
input_dir: datasets/test/images
output_csv: batch_predictions.csv
format: png
crop: 1024
zoom: false
zoom_factor: 0.667
```

### annotate_images_config.yaml
```yaml
model: yolov8s-seg_yfusion.pt
input: images_CNN_clean/
output: annotation_results.csv
format: auto
confidence: 0.5
imgsz: 1024
zoom: false
verbose: false
```

## Common Use Cases

### High-Resolution Processing
```bash
# Edit config: zoom: true, zoom_factor: 0.5
python batch_predict.py --config batch_predict_config.yaml
```

### Low-Confidence Exploration
```bash
python annotate_images.py --config annotate_images_config.yaml --confidence 0.3
```

### Quick Override
```bash
# Process different directory
python batch_predict.py --config batch_predict_config.yaml \
  --input_dir new_data/ --output_csv new_results.csv
```

## Testing

```bash
# Test configs are valid
python tests/test_config_files.py

# Test output format
python tests/test_prediction_outputs.py

# Run all tests
python tests/run_tests.py
```

## Documentation

- **CONFIG_GUIDE.md** - Detailed configuration guide
- **CONFIGURATION_SUMMARY.md** - Implementation summary
- **tests/README.md** - Testing documentation

## Created Files

✓ batch_predict_config.yaml
✓ annotate_images_config.yaml
✓ CONFIG_GUIDE.md
✓ CONFIGURATION_SUMMARY.md
✓ tests/test_config_files.py
✓ Modified: batch_predict.py (config support)
✓ Modified: annotate_images.py (config support)
✓ Modified: tests/README.md (config docs)
