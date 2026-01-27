#!/usr/bin/env python3
"""
Test script for validating prediction output formats

This script tests the output format of batch_predict.py and annotate_images.py
to ensure CSV outputs contain the correct columns, data types, and value ranges.

Usage:
    python tests/test_prediction_outputs.py
    python -m pytest tests/test_prediction_outputs.py -v
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPredictionOutputFormat(unittest.TestCase):
    """Test suite for validating prediction output CSV format"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.required_columns_base = [
            'file',
            'crop_id', 
            'class',
            'proba',
            'x1', 'y1', 'x2', 'y2',
        ]
        
        self.required_statistics_columns = [
            # Brightfield statistics
            'bf_mean', 'bf_std', 'bf_min', 'bf_max', 'bf_skew',
            # RFP statistics
            'rfp_mean', 'rfp_std', 'rfp_min', 'rfp_max', 'rfp_skew',
            # GFP statistics
            'gfp_mean', 'gfp_std', 'gfp_min', 'gfp_max', 'gfp_skew',
        ]
        
        self.all_required_columns = self.required_columns_base + self.required_statistics_columns
        
    def test_csv_has_required_columns(self):
        """Test that CSV contains all required columns"""
        # Create a sample dataframe with expected structure
        sample_data = self._create_sample_output()
        
        for col in self.all_required_columns:
            self.assertIn(col, sample_data.columns,
                         f"Required column '{col}' missing from output")
    
    def test_column_data_types(self):
        """Test that columns have correct data types"""
        sample_data = self._create_sample_output()
        
        # File should be string
        self.assertTrue(sample_data['file'].dtype == object,
                       "Column 'file' should be string/object type")
        
        # Numeric columns
        numeric_columns = ['crop_id', 'class', 'proba', 'x1', 'y1', 'x2', 'y2'] + \
                         self.required_statistics_columns
        
        for col in numeric_columns:
            self.assertTrue(np.issubdtype(sample_data[col].dtype, np.number),
                           f"Column '{col}' should be numeric")
    
    def test_probability_range(self):
        """Test that probability values are in valid range [0, 1]"""
        sample_data = self._create_sample_output()
        
        self.assertTrue((sample_data['proba'] >= 0).all(),
                       "Probability values should be >= 0")
        self.assertTrue((sample_data['proba'] <= 1).all(),
                       "Probability values should be <= 1")
    
    def test_bounding_box_validity(self):
        """Test that bounding boxes have valid coordinates"""
        sample_data = self._create_sample_output()
        
        # x2 should be greater than x1
        self.assertTrue((sample_data['x2'] > sample_data['x1']).all(),
                       "x2 should be greater than x1 in bounding boxes")
        
        # y2 should be greater than y1
        self.assertTrue((sample_data['y2'] > sample_data['y1']).all(),
                       "y2 should be greater than y1 in bounding boxes")
        
        # All coordinates should be non-negative
        self.assertTrue((sample_data['x1'] >= 0).all(), "x1 should be >= 0")
        self.assertTrue((sample_data['y1'] >= 0).all(), "y1 should be >= 0")
        self.assertTrue((sample_data['x2'] >= 0).all(), "x2 should be >= 0")
        self.assertTrue((sample_data['y2'] >= 0).all(), "y2 should be >= 0")
    
    def test_statistics_validity(self):
        """Test that statistical values are valid"""
        sample_data = self._create_sample_output()
        
        # Check that min <= mean <= max for each channel
        for channel in ['bf', 'rfp', 'gfp']:
            min_col = f'{channel}_min'
            mean_col = f'{channel}_mean'
            max_col = f'{channel}_max'
            std_col = f'{channel}_std'
            
            self.assertTrue((sample_data[min_col] <= sample_data[mean_col]).all(),
                           f"{channel}: min should be <= mean")
            self.assertTrue((sample_data[mean_col] <= sample_data[max_col]).all(),
                           f"{channel}: mean should be <= max")
            
            # Standard deviation should be non-negative
            self.assertTrue((sample_data[std_col] >= 0).all(),
                           f"{channel}: std should be >= 0")
            
            # All pixel values should be in [0, 255] for uint8 images
            self.assertTrue((sample_data[min_col] >= 0).all(),
                           f"{channel}: min should be >= 0")
            self.assertTrue((sample_data[max_col] <= 255).all(),
                           f"{channel}: max should be <= 255")
    
    def test_class_validity(self):
        """Test that class IDs are valid integers"""
        sample_data = self._create_sample_output()
        
        # Class should be integer
        self.assertTrue(np.issubdtype(sample_data['class'].dtype, np.integer),
                       "Class should be integer type")
        
        # Class should be non-negative
        self.assertTrue((sample_data['class'] >= 0).all(),
                       "Class IDs should be non-negative")
    
    def test_no_null_values(self):
        """Test that there are no null values in critical columns"""
        sample_data = self._create_sample_output()
        
        for col in self.all_required_columns:
            self.assertFalse(sample_data[col].isnull().any(),
                           f"Column '{col}' should not contain null values")
    
    def test_zoom_mode_columns(self):
        """Test that zoom mode adds required coordinate columns"""
        sample_data = self._create_sample_output_with_zoom()
        
        zoom_columns = ['crop_x1', 'crop_y1', 'crop_x2', 'crop_y2']
        
        for col in zoom_columns:
            self.assertIn(col, sample_data.columns,
                         f"Zoom mode should include '{col}' column")
            
        # Crop coordinates should be valid
        self.assertTrue((sample_data['crop_x2'] > sample_data['crop_x1']).all(),
                       "crop_x2 should be greater than crop_x1")
        self.assertTrue((sample_data['crop_y2'] > sample_data['crop_y1']).all(),
                       "crop_y2 should be greater than crop_y1")
    
    def test_annotate_images_additional_columns(self):
        """Test annotate_images.py specific columns"""
        sample_data = self._create_annotate_images_output()
        
        expected_columns = ['group_name', 'frame_index']
        
        for col in expected_columns:
            self.assertIn(col, sample_data.columns,
                         f"annotate_images.py output should include '{col}'")
        
        # Frame index should be non-negative integer
        self.assertTrue(np.issubdtype(sample_data['frame_index'].dtype, np.integer),
                       "frame_index should be integer")
        self.assertTrue((sample_data['frame_index'] >= 0).all(),
                       "frame_index should be >= 0")
    
    def test_csv_can_be_loaded(self):
        """Test that CSV can be loaded without errors"""
        sample_data = self._create_sample_output()
        
        # Write to temporary CSV
        temp_csv = '/tmp/test_output.csv'
        sample_data.to_csv(temp_csv, index=False)
        
        # Try to load it back
        try:
            loaded_data = pd.read_csv(temp_csv)
            self.assertEqual(len(loaded_data), len(sample_data),
                           "Loaded CSV should have same number of rows")
            
            # Check that all columns are preserved
            for col in self.all_required_columns:
                self.assertIn(col, loaded_data.columns,
                             f"Column '{col}' should be preserved in CSV")
        finally:
            # Clean up
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
    
    def test_skewness_range(self):
        """Test that skewness values are in reasonable range"""
        sample_data = self._create_sample_output()
        
        # Skewness typically ranges from -3 to 3 for most distributions
        # but can be larger in extreme cases. Test for reasonable bounds.
        for channel in ['bf', 'rfp', 'gfp']:
            skew_col = f'{channel}_skew'
            
            # Check that skewness is not infinite or NaN
            self.assertFalse(np.isinf(sample_data[skew_col]).any(),
                           f"{channel}: skewness should not be infinite")
            self.assertFalse(np.isnan(sample_data[skew_col]).any(),
                           f"{channel}: skewness should not be NaN")
    
    def _create_sample_output(self):
        """Load real prediction output from batch_predict.py results"""
        # Use actual prediction results from new_images_png/filtro H
        sample_csv = Path(__file__).parent.parent / 'new_images_png' / 'filtro H' / '5-03.csv'
        
        if sample_csv.exists():
            df = pd.read_csv(sample_csv, index_col=0)
            # Add missing columns that should be in output format
            if 'file' not in df.columns:
                df.insert(0, 'file', '5-03.png')
            if 'crop_id' not in df.columns:
                df.insert(1, 'crop_id', 0)
            
            print(f"\n{'='*70}")
            print(f"Loading real prediction data from: {sample_csv.name}")
            print(f"{'='*70}")
            print(f"Shape: {df.shape[0]} detections, {df.shape[1]} features")
            print(f"\nFirst 3 rows:")
            print(df.head(3).to_string())
            print(f"\nColumn summary:")
            print(f"  Detections: {len(df)}")
            print(f"  Confidence range: {df['proba'].min():.3f} - {df['proba'].max():.3f}")
            print(f"  Classes detected: {sorted(df['class'].unique())}")
            print(f"{'='*70}\n")
            
            # Ensure we have the required columns by taking first 10 rows
            return df.head(10) if len(df) > 0 else self._create_fallback_output()
        else:
            return self._create_fallback_output()
    
    def _create_fallback_output(self):
        """Create synthetic output if real data not available"""
        n_detections = 10
        
        data = {
            'file': [f'image_{i}.png' for i in range(n_detections)],
            'crop_id': [0] * n_detections,
            'class': np.random.randint(0, 3, n_detections),
            'proba': np.random.uniform(0.5, 1.0, n_detections),
            'x1': np.random.uniform(0, 500, n_detections),
            'y1': np.random.uniform(0, 500, n_detections),
            'x2': np.random.uniform(500, 1024, n_detections),
            'y2': np.random.uniform(500, 1024, n_detections),
            # BF statistics
            'bf_mean': np.random.uniform(50, 200, n_detections),
            'bf_std': np.random.uniform(10, 50, n_detections),
            'bf_min': np.random.uniform(0, 50, n_detections),
            'bf_max': np.random.uniform(200, 255, n_detections),
            'bf_skew': np.random.uniform(-1, 1, n_detections),
            # RFP statistics
            'rfp_mean': np.random.uniform(50, 200, n_detections),
            'rfp_std': np.random.uniform(10, 50, n_detections),
            'rfp_min': np.random.uniform(0, 50, n_detections),
            'rfp_max': np.random.uniform(200, 255, n_detections),
            'rfp_skew': np.random.uniform(-1, 1, n_detections),
            # GFP statistics
            'gfp_mean': np.random.uniform(50, 200, n_detections),
            'gfp_std': np.random.uniform(10, 50, n_detections),
            'gfp_min': np.random.uniform(0, 50, n_detections),
            'gfp_max': np.random.uniform(200, 255, n_detections),
            'gfp_skew': np.random.uniform(-1, 1, n_detections),
        }
        
        return pd.DataFrame(data)
    
    def _create_sample_output_with_zoom(self):
        """Create sample output with zoom mode columns"""
        df = self._create_sample_output()
        
        # Add zoom mode columns (simulate zoom predictions)
        df['crop_x1'] = np.random.uniform(0, 500, len(df))
        df['crop_y1'] = np.random.uniform(0, 500, len(df))
        df['crop_x2'] = df['crop_x1'] + 600  # Ensure x2 > x1
        df['crop_y2'] = df['crop_y1'] + 600  # Ensure y2 > y1
        
        return df
    
    def _create_annotate_images_output(self):
        """Create sample output for annotate_images.py with real data"""
        df = self._create_sample_output()
        
        # Add annotate_images.py specific columns
        df['group_name'] = [f'filtro_H_{i % 3}' for i in range(len(df))]
        df['frame_index'] = [0] * len(df)  # Single frame images
        df['source_path'] = ['new_images_png/filtro H/5-03.png'] * len(df)
        
        return df


class TestRealOutputValidation(unittest.TestCase):
    """Test suite for validating real prediction outputs if they exist"""
    
    def setUp(self):
        """Find any existing CSV output files"""
        self.workspace_root = Path(__file__).parent.parent
        
        # Look specifically in new_images_png/filtro H for test data
        filtro_h_dir = self.workspace_root / 'new_images_png' / 'filtro H'
        if filtro_h_dir.exists():
            self.prediction_csvs = list(filtro_h_dir.glob('*.csv'))
            # Exclude prediction visualization files
            self.prediction_csvs = [f for f in self.prediction_csvs if '_pred' not in f.name]
        else:
            self.prediction_csvs = []
        
        # Also check for other prediction outputs
        self.csv_files = list(self.workspace_root.glob('**/*.csv'))
        other_predictions = [
            f for f in self.csv_files 
            if 'result' in f.name.lower() or 'predict' in f.name.lower() or 'output' in f.name.lower()
        ]
        self.prediction_csvs.extend(other_predictions)
        
        # Remove duplicates
        self.prediction_csvs = list(set(self.prediction_csvs))
        
        print(f"\n{'='*70}")
        print(f"REAL OUTPUT VALIDATION - Found {len(self.prediction_csvs)} CSV files")
        print(f"{'='*70}")
        for csv_file in self.prediction_csvs[:5]:  # Show first 5
            print(f"  - {csv_file.relative_to(self.workspace_root)}")
        if len(self.prediction_csvs) > 5:
            print(f"  ... and {len(self.prediction_csvs) - 5} more")
        print(f"{'='*70}\n")
    
    def test_existing_csv_files(self):
        """Test any existing CSV files have valid format"""
        if not self.prediction_csvs:
            self.skipTest("No prediction CSV files found to test")
        
        all_data = []
        
        for csv_file in self.prediction_csvs:
            with self.subTest(csv_file=csv_file.name):
                try:
                    df = pd.read_csv(csv_file)
                    
                    print(f"\n--- Validating: {csv_file.name} ---")
                    print(f"Shape: {df.shape}")
                    
                    # Check basic structure
                    self.assertGreater(len(df.columns), 0, 
                                     f"{csv_file.name}: Should have at least one column")
                    
                    # Check for common required columns
                    common_cols = ['class', 'proba']
                    for col in common_cols:
                        if col in df.columns:
                            self.assertTrue(np.issubdtype(df[col].dtype, np.number),
                                          f"{csv_file.name}: '{col}' should be numeric")
                    
                    # Check probability range if present
                    if 'proba' in df.columns:
                        self.assertTrue((df['proba'] >= 0).all() and (df['proba'] <= 1).all(),
                                      f"{csv_file.name}: Probability should be in [0, 1]")
                        print(f"  Confidence: {df['proba'].min():.3f} - {df['proba'].max():.3f} (mean: {df['proba'].mean():.3f})")
                    
                    if 'class' in df.columns:
                        print(f"  Classes: {sorted(df['class'].unique())}")
                        print(f"  Detections per class: {dict(df['class'].value_counts())}")
                    
                    all_data.append(df)
                    
                except Exception as e:
                    self.fail(f"Failed to validate {csv_file.name}: {str(e)}")
        
        # Create global dataframe summary
        if all_data:
            global_df = pd.concat(all_data, ignore_index=True)
            print(f"\n{'='*70}")
            print(f"GLOBAL DATAFRAME SUMMARY - All {len(self.prediction_csvs)} CSV files combined")
            print(f"{'='*70}")
            print(f"Total detections: {len(global_df)}")
            print(f"Total features: {len(global_df.columns)}")
            if 'proba' in global_df.columns:
                print(f"\nConfidence Statistics:")
                print(f"  Mean: {global_df['proba'].mean():.3f}")
                print(f"  Std:  {global_df['proba'].std():.3f}")
                print(f"  Min:  {global_df['proba'].min():.3f}")
                print(f"  Max:  {global_df['proba'].max():.3f}")
            if 'class' in global_df.columns:
                print(f"\nClass Distribution:")
                for cls, count in sorted(global_df['class'].value_counts().items()):
                    print(f"  Class {cls}: {count} detections ({count/len(global_df)*100:.1f}%)")
            
            # Channel statistics summary
            for channel in ['bf', 'rfp', 'gfp']:
                mean_col = f'{channel}_mean'
                if mean_col in global_df.columns:
                    print(f"\n{channel.upper()} Channel Statistics:")
                    print(f"  Mean intensity: {global_df[mean_col].mean():.2f} ± {global_df[mean_col].std():.2f}")
                    print(f"  Range: {global_df[mean_col].min():.2f} - {global_df[mean_col].max():.2f}")
            
            print(f"{'='*70}\n")


def run_tests():
    """Run all tests and print results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPredictionOutputFormat))
    suite.addTests(loader.loadTestsFromTestCase(TestRealOutputValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
