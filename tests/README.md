# Tests for Yeast Fusion Segmenter

This directory contains comprehensive tests for the Yeast Fusion Segmenter prediction pipeline.

## Test Files

### `test_prediction_outputs.py`
Tests the output format of prediction scripts to ensure CSV files contain correct columns, data types, and value ranges.

**Key Tests:**
- CSV column validation (required columns present)
- Data type validation (numeric, string types)
- Probability range validation (0-1)
- Bounding box validity (x2 > x1, y2 > y1, non-negative coordinates)
- Statistical value validity (min ≤ mean ≤ max, non-negative std)
- Class ID validation (non-negative integers)
- Null value checks
- Zoom mode coordinate validation
- Skewness range validation

**Output Format Specification:**

Required columns for `batch_predict.py` and `annotate_images.py`:
```
file             - String: Image filename
crop_id          - Integer: Crop identifier (0 for full image)
class            - Integer: Class ID (≥ 0)
proba            - Float: Confidence score (0-1)
x1, y1, x2, y2   - Float: Bounding box coordinates (x2>x1, y2>y1, all ≥ 0)
bf_mean          - Float: Brightfield mean intensity (0-255)
bf_std           - Float: Brightfield variance (≥ 0)
bf_min           - Float: Brightfield minimum (0-255)
bf_max           - Float: Brightfield maximum (0-255)
bf_skew          - Float: Brightfield skewness
rfp_mean         - Float: RFP mean intensity (0-255)
rfp_std          - Float: RFP variance (≥ 0)
rfp_min          - Float: RFP minimum (0-255)
rfp_max          - Float: RFP maximum (0-255)
rfp_skew         - Float: RFP skewness
gfp_mean         - Float: GFP mean intensity (0-255)
gfp_std          - Float: GFP variance (≥ 0)
gfp_min          - Float: GFP minimum (0-255)
gfp_max          - Float: GFP maximum (0-255)
gfp_skew         - Float: GFP skewness
```

Additional columns when using `--zoom` mode:
```
crop_x1, crop_y1 - Float: Top-left coordinates of crop in original image
crop_x2, crop_y2 - Float: Bottom-right coordinates of crop in original image
```

Additional columns for `annotate_images.py`:
```
group_name       - String: Image group identifier
frame_index      - Integer: Frame number (≥ 0)
source_path      - String: Path to source image file
```

### `test_integration.py`
Integration tests for the prediction pipeline, including actual function testing.

**Key Tests:**
- Module import validation
- Image processing function tests
- zoom_img() crop generation and coordinate validation
- yield_frames() TIFF processing and normalization
- process_image() format handling
- CSV structure consistency
- Empty prediction handling

### `test_config_files.py`
Tests for YAML configuration file support in prediction scripts.

**Key Tests:**
- Config file existence validation
- YAML structure and required fields
- Parameter value ranges (confidence, crop size, etc.)
- Model specification (yolov8s-seg_yfusion.pt)
- Config file loading and parsing
- Field preservation after load/save

## Running Tests

### Run all tests:
```bash
# Using unittest
python tests/test_prediction_outputs.py
python tests/test_integration.py
python tests/test_config_files.py

# Using pytest (if installed)
pytest tests/ -v
pytest tests/test_prediction_outputs.py -v
pytest tests/test_integration.py -v
pytest tests/test_config_files.py -v
```

### Run specific test class:
```bash
python -m pytest tests/test_prediction_outputs.py::TestPredictionOutputFormat -v
python -m pytest tests/test_integration.py::TestBatchPredictIntegration -v
```

### Run with coverage (if pytest-cov installed):
```bash
pytest tests/ --cov=. --cov-report=html
```

## Test Requirements

Install test dependencies:
```bash
pip install pytest pytest-cov
```

Or use existing project requirements:
```bash
pip install -r requirements.txt
```

## Expected Output

When tests pass successfully:
```
======================================================================
TEST SUMMARY
======================================================================
Tests run: 15
Successes: 15
Failures: 0
Errors: 0
Skipped: 0
======================================================================
```

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
```

## Validation Guidelines

### For batch_predict.py outputs:
1. All detections must have confidence > 0.5
2. Bounding boxes must be valid (x2 > x1, y2 > y1)
3. Statistical values must be in range [0, 255] for pixel intensities
4. Each detection must have all 15 statistical features (5 per channel)

### For annotate_images.py outputs:
1. All requirements from batch_predict.py apply
2. Must include group_name and frame_index
3. Multiple frames from same image must share same group_name
4. frame_index must be sequential starting from 0

## Troubleshooting

### Common Issues:

**ImportError: No module named 'batch_predict'**
- Ensure you're running from project root
- Tests automatically add parent directory to path

**FileNotFoundError during tests**
- Integration tests create temporary files automatically
- Ensure write permissions in /tmp directory

**Test failures on real CSV files**
- Check that CSV files match expected format
- Verify all required columns are present
- Ensure data types are correct

## Configuration File Usage

Both prediction scripts now support YAML configuration files:

### batch_predict.py with config:
```bash
python batch_predict.py --config batch_predict_config.yaml
```

### annotate_images.py with config:
```bash
python annotate_images.py --config annotate_images_config.yaml
```

### Override config values:
```bash
# Use config file but override specific parameters
python batch_predict.py --config batch_predict_config.yaml --crop 512 --zoom
python annotate_images.py --config annotate_images_config.yaml --confidence 0.7
```

## Contributing

When adding new features to prediction scripts:
1. Update test cases to cover new functionality
2. Add expected columns to format specification
3. Update config file templates if new parameters added
4. Ensure backward compatibility with existing outputs
5. Run all tests before committing changes
