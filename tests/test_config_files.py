#!/usr/bin/env python3
"""
Test configuration file support for prediction scripts

This module tests that both batch_predict.py and annotate_images.py
can properly load and use YAML configuration files.
"""

import unittest
import os
import sys
import tempfile
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfigFileSupport(unittest.TestCase):
    """Test YAML configuration file support"""
    
    def setUp(self):
        """Create temporary config files for testing"""
        self.test_dir = tempfile.mkdtemp(prefix='config_test_')
        
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_batch_predict_config_exists(self):
        """Test that batch_predict_config.yaml exists and is valid"""
        config_path = Path(__file__).parent.parent / 'batch_predict_config.yaml'
        
        self.assertTrue(config_path.exists(), 
                       "batch_predict_config.yaml should exist")
        
        # Load and validate structure
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['model', 'input_dir', 'output_csv', 'format']
        for field in required_fields:
            self.assertIn(field, config,
                         f"Config should contain '{field}' field")
    
    def test_annotate_images_config_exists(self):
        """Test that annotate_images_config.yaml exists and is valid"""
        config_path = Path(__file__).parent.parent / 'annotate_images_config.yaml'
        
        self.assertTrue(config_path.exists(),
                       "annotate_images_config.yaml should exist")
        
        # Load and validate structure
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['model', 'input', 'output']
        for field in required_fields:
            self.assertIn(field, config,
                         f"Config should contain '{field}' field")
    
    def test_batch_predict_config_values(self):
        """Test that batch_predict config has valid values"""
        config_path = Path(__file__).parent.parent / 'batch_predict_config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check model path format
        self.assertTrue(config['model'].endswith('.pt'),
                       "Model should be a .pt file")
        
        # Check format is valid
        self.assertIn(config['format'], ['png', 'tif', 'czi'],
                     "Format should be png, tif, or czi")
        
        # Check crop size is reasonable
        self.assertGreater(config['crop'], 0,
                          "Crop size should be positive")
        self.assertLessEqual(config['crop'], 4096,
                            "Crop size should be reasonable")
        
        # Check zoom factor
        self.assertGreater(config['zoom_factor'], 0,
                          "Zoom factor should be positive")
        self.assertLessEqual(config['zoom_factor'], 1.0,
                            "Zoom factor should be <= 1.0")
    
    def test_annotate_images_config_values(self):
        """Test that annotate_images config has valid values"""
        config_path = Path(__file__).parent.parent / 'annotate_images_config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check model path format
        self.assertTrue(config['model'].endswith('.pt'),
                       "Model should be a .pt file")
        
        # Check confidence threshold
        self.assertGreaterEqual(config['confidence'], 0.0,
                               "Confidence should be >= 0.0")
        self.assertLessEqual(config['confidence'], 1.0,
                            "Confidence should be <= 1.0")
        
        # Check image size
        self.assertGreater(config['imgsz'], 0,
                          "Image size should be positive")
        
        # Check format
        self.assertIn(config['format'], ['auto', 'tiff', 'czi', 'single'],
                     "Format should be valid option")
    
    def test_config_file_loading_function(self):
        """Test that config files can be loaded properly"""
        # Create a test config
        test_config = {
            'model': 'test_model.pt',
            'input_dir': 'test_input',
            'output_csv': 'test_output.csv',
            'format': 'png',
            'crop': 512
        }
        
        test_config_path = os.path.join(self.test_dir, 'test_config.yaml')
        
        with open(test_config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Load it back
        with open(test_config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        # Verify all fields preserved
        for key, value in test_config.items():
            self.assertEqual(loaded_config[key], value,
                           f"Config field '{key}' should be preserved")
    
    def test_config_uses_yolov8s_model(self):
        """Test that both configs use the yolov8s-seg_yfusion.pt model"""
        # Check batch_predict config
        with open(Path(__file__).parent.parent / 'batch_predict_config.yaml', 'r') as f:
            batch_config = yaml.safe_load(f)
        
        self.assertEqual(batch_config['model'], 'yolov8s-seg_yfusion.pt',
                        "batch_predict should use yolov8s-seg_yfusion.pt model")
        
        # Check annotate_images config
        with open(Path(__file__).parent.parent / 'annotate_images_config.yaml', 'r') as f:
            annotate_config = yaml.safe_load(f)
        
        self.assertEqual(annotate_config['model'], 'yolov8s-seg_yfusion.pt',
                        "annotate_images should use yolov8s-seg_yfusion.pt model")


def run_config_tests():
    """Run configuration file tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestConfigFileSupport))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("CONFIG FILE TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_config_tests()
    sys.exit(0 if success else 1)
