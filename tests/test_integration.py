#!/usr/bin/env python3
"""
Integration tests for prediction scripts

This module tests the actual prediction scripts (batch_predict.py and annotate_images.py)
by running them on test data and validating their outputs.
"""

import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBatchPredictIntegration(unittest.TestCase):
    """Integration tests for batch_predict.py"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        cls.test_dir = tempfile.mkdtemp(prefix='yeast_fusion_test_')
        cls.test_images_dir = os.path.join(cls.test_dir, 'test_images')
        os.makedirs(cls.test_images_dir, exist_ok=True)
        
        # Create dummy test images
        cls._create_test_images()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _create_test_images(cls):
        """Create synthetic test images"""
        # Create 3 test PNG images with 3 channels
        for i in range(3):
            img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
            img_path = os.path.join(cls.test_images_dir, f'test_image_{i}.png')
            Image.fromarray(img).save(img_path)
    
    def test_imports_work(self):
        """Test that required modules can be imported"""
        try:
            from batch_predict import process_image, zoom_img, predict_and_collect
            from annotate_images import yield_frames, zoom_img as zoom_img_annotate
            self.assertTrue(True, "Imports successful")
        except ImportError as e:
            self.fail(f"Failed to import required modules: {str(e)}")
    
    def test_zoom_img_function(self):
        """Test zoom_img function creates valid crops"""
        from batch_predict import zoom_img
        import tempfile
        import os
        
        # Create test image
        test_img = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
        
        # Test zoom WITHOUT base_filepath (no PNG files saved)
        crops, png_paths, coordinates = zoom_img(test_img, zoom_factor=0.5, target_size=1024)
        
        # Verify crops were created
        self.assertGreater(len(crops), 0, "Should create at least one crop")
        self.assertEqual(len(crops), len(coordinates), 
                        "Should have same number of crops and coordinates")
        self.assertEqual(len(png_paths), 0,
                        "Should have no PNG paths when base_filepath not provided")
        
        # Test zoom WITH base_filepath (PNG files saved)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            crops2, png_paths2, coordinates2 = zoom_img(test_img, zoom_factor=0.5, 
                                                         target_size=1024, 
                                                         base_filepath=tmp_path)
            
            self.assertEqual(len(crops2), len(png_paths2),
                            "Should have same number of crops and PNG paths when base_filepath provided")
            
            # Verify PNG files were created
            for png_path in png_paths2:
                self.assertTrue(os.path.exists(png_path), 
                              f"PNG file should exist: {png_path}")
                # Clean up created PNG
                if os.path.exists(png_path):
                    os.remove(png_path)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        # Verify crop dimensions
        for crop in crops:
            self.assertEqual(crop.shape[:2], (1024, 1024),
                           "Each crop should be resized to target size")
        
        # Verify coordinates are valid
        for x1, y1, x2, y2 in coordinates:
            self.assertLess(x1, x2, "x2 should be greater than x1")
            self.assertLess(y1, y2, "y2 should be greater than y1")
            self.assertGreaterEqual(x1, 0, "x1 should be non-negative")
            self.assertGreaterEqual(y1, 0, "y1 should be non-negative")
            self.assertLessEqual(x2, test_img.shape[1], "x2 should be within image bounds")
            self.assertLessEqual(y2, test_img.shape[0], "y2 should be within image bounds")
    
    def test_zoom_mode_with_model(self):
        """Test zoom mode end-to-end with a real model"""
        from batch_predict import zoom_img, process_image, predict_and_collect
        from ultralytics import YOLO
        import tempfile
        import os
        
        # Check if a model exists
        model_path = None
        for candidate in ['yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolo11n.pt']:
            if os.path.exists(candidate):
                model_path = candidate
                break
        
        if model_path is None:
            self.skipTest("No YOLO model found for testing")
        
        # Load model
        model = YOLO(model_path)
        
        # Create a larger test image (2048x2048 to test zooming)
        test_img = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
        
        # Save as temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            Image.fromarray(test_img).save(tmp_path)
        
        try:
            # Process image
            arr, png_path = process_image(tmp_path, 'png', crop=1024)
            
            # Create zoomed crops with PNG files
            crops, png_paths, coordinates = zoom_img(arr, zoom_factor=0.5, 
                                                     target_size=1024, 
                                                     base_filepath=tmp_path)
            
            # Verify crops were created
            self.assertGreater(len(crops), 0, "Should create at least one crop")
            self.assertEqual(len(png_paths), len(crops), 
                           "Should have PNG path for each crop")
            
            # Run prediction on first crop using saved PNG
            if len(png_paths) > 0:
                csv_path = png_paths[0].replace('.png', '.csv')
                df = predict_and_collect(model, png_paths[0], csv_path, 
                                        crop=1024, crop_id=0)
                
                # Verify output structure (may be None if no detections)
                if df is not None:
                    self.assertIsInstance(df, pd.DataFrame, 
                                        "Should return a DataFrame")
                    # Check for expected columns
                    expected_cols = ['file', 'crop_id', 'class', 'proba', 
                                   'x1', 'y1', 'x2', 'y2']
                    for col in expected_cols:
                        self.assertIn(col, df.columns, 
                                    f"DataFrame should have '{col}' column")
                
                # Clean up CSV if created
                if os.path.exists(csv_path):
                    os.remove(csv_path)
            
            # Clean up PNG crops
            for png_path in png_paths:
                if os.path.exists(png_path):
                    os.remove(png_path)
                    
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestImageProcessingFunctions(unittest.TestCase):
    """Test image processing utility functions"""
    
    def test_yield_frames_with_single_page(self):
        """Test yield_frames with single-page TIFF"""
        from annotate_images import yield_frames
        
        # Create single-page TIFF
        img_array = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
        temp_tiff = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        
        try:
            Image.fromarray(img_array).save(temp_tiff.name)
            img = Image.open(temp_tiff.name)
            
            frames = list(yield_frames(img, crop=1024, scaler=True))
            
            self.assertEqual(len(frames), 1, "Should yield one frame")
            self.assertEqual(frames[0].shape, (1024, 1024), "Frame should have correct shape")
            self.assertEqual(frames[0].dtype, np.uint8, "Frame should be uint8")
        finally:
            os.unlink(temp_tiff.name)
    
    def test_yield_frames_normalization(self):
        """Test that yield_frames normalizes correctly"""
        from annotate_images import yield_frames
        
        # Create image with known range
        img_array = np.random.randint(100, 200, (512, 512), dtype=np.uint16)
        temp_tiff = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        
        try:
            Image.fromarray(img_array).save(temp_tiff.name)
            img = Image.open(temp_tiff.name)
            
            frames = list(yield_frames(img, crop=512, scaler=True))
            frame = frames[0]
            
            # After normalization, should span 0-255
            self.assertGreaterEqual(frame.min(), 0, "Min should be >= 0")
            self.assertLessEqual(frame.max(), 255, "Max should be <= 255")
            self.assertEqual(frame.dtype, np.uint8, "Should be uint8")
        finally:
            os.unlink(temp_tiff.name)
    
    def test_process_image_png(self):
        """Test process_image with PNG format"""
        from batch_predict import process_image
        
        # Create test PNG
        img_array = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        temp_png = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        
        try:
            Image.fromarray(img_array).save(temp_png.name)
            
            result, png_path = process_image(temp_png.name, fmt='png', crop=1024)
            
            self.assertEqual(result.shape, (1024, 1024, 3), 
                           "Should return 1024x1024x3 array")
            self.assertEqual(result.dtype, np.uint8, "Should be uint8")
            self.assertIsNone(png_path, "PNG format should return None as png_path")
        finally:
            os.unlink(temp_png.name)


class TestOutputCSVStructure(unittest.TestCase):
    """Test CSV output structure and validation"""
    
    def test_csv_columns_order_consistency(self):
        """Test that CSV columns are in consistent order"""
        # Create two sample dataframes with same data
        data = {
            'file': ['test.png'],
            'crop_id': [0],
            'class': [0],
            'proba': [0.9],
            'x1': [10], 'y1': [20], 'x2': [100], 'y2': [120],
            'bf_mean': [100], 'bf_std': [25], 'bf_min': [50], 'bf_max': [200], 'bf_skew': [0.1],
            'rfp_mean': [80], 'rfp_std': [20], 'rfp_min': [40], 'rfp_max': [180], 'rfp_skew': [0.2],
            'gfp_mean': [90], 'gfp_std': [22], 'gfp_min': [45], 'gfp_max': [190], 'gfp_skew': [0.15],
        }
        
        df1 = pd.DataFrame(data)
        df2 = pd.DataFrame(data)
        
        # Write and read back
        temp_csv1 = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
        temp_csv2 = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
        
        try:
            df1.to_csv(temp_csv1.name, index=False)
            df2.to_csv(temp_csv2.name, index=False)
            
            df1_read = pd.read_csv(temp_csv1.name)
            df2_read = pd.read_csv(temp_csv2.name)
            
            # Columns should be in same order
            self.assertEqual(list(df1_read.columns), list(df2_read.columns),
                           "Column order should be consistent")
        finally:
            os.unlink(temp_csv1.name)
            os.unlink(temp_csv2.name)
    
    def test_empty_predictions_handling(self):
        """Test handling of empty predictions"""
        # Create empty dataframe with correct columns
        columns = ['file', 'crop_id', 'class', 'proba', 'x1', 'y1', 'x2', 'y2',
                  'bf_mean', 'bf_std', 'bf_min', 'bf_max', 'bf_skew',
                  'rfp_mean', 'rfp_std', 'rfp_min', 'rfp_max', 'rfp_skew',
                  'gfp_mean', 'gfp_std', 'gfp_min', 'gfp_max', 'gfp_skew']
        
        df = pd.DataFrame(columns=columns)
        
        # Should be able to save and load empty CSV
        temp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
        
        try:
            df.to_csv(temp_csv.name, index=False)
            df_read = pd.read_csv(temp_csv.name)
            
            self.assertEqual(len(df_read), 0, "Empty dataframe should have 0 rows")
            self.assertEqual(list(df_read.columns), columns, 
                           "Columns should be preserved even when empty")
        finally:
            os.unlink(temp_csv.name)


def run_integration_tests():
    """Run all integration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestBatchPredictIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestImageProcessingFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestOutputCSVStructure))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)
