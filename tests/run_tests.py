#!/usr/bin/env python3
"""
Test runner script for Yeast Fusion Segmenter

This script runs all tests and provides a summary report.
Can be used for quick validation or CI/CD pipelines.

Usage:
    python tests/run_tests.py
    python tests/run_tests.py --verbose
    python tests/run_tests.py --integration-only
    python tests/run_tests.py --format-only
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest


def run_all_tests(verbose=1, integration_only=False, format_only=False):
    """
    Run all tests and return results
    
    Args:
        verbose: Verbosity level (0, 1, or 2)
        integration_only: Only run integration tests
        format_only: Only run format validation tests
        
    Returns:
        bool: True if all tests passed
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Load test modules
    if not integration_only:
        try:
            from test_prediction_outputs import TestPredictionOutputFormat, TestRealOutputValidation
            suite.addTests(loader.loadTestsFromTestCase(TestPredictionOutputFormat))
            suite.addTests(loader.loadTestsFromTestCase(TestRealOutputValidation))
            print("✓ Loaded format validation tests")
        except ImportError as e:
            print(f"✗ Failed to load format tests: {e}")
    
    if not format_only:
        try:
            from test_integration import (TestBatchPredictIntegration, 
                                        TestImageProcessingFunctions,
                                        TestOutputCSVStructure)
            suite.addTests(loader.loadTestsFromTestCase(TestBatchPredictIntegration))
            suite.addTests(loader.loadTestsFromTestCase(TestImageProcessingFunctions))
            suite.addTests(loader.loadTestsFromTestCase(TestOutputCSVStructure))
            print("✓ Loaded integration tests")
        except ImportError as e:
            print(f"✗ Failed to load integration tests: {e}")
    
    # Run tests
    print("\n" + "="*70)
    print("RUNNING TESTS")
    print("="*70 + "\n")
    
    runner = unittest.TextTestRunner(verbosity=verbose)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run:     {result.testsRun}")
    print(f"✓ Successes:   {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"✗ Failures:    {len(result.failures)}")
    print(f"✗ Errors:      {len(result.errors)}")
    print(f"⊘ Skipped:     {len(result.skipped)}")
    print("="*70)
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!\n")
    else:
        print("\n✗ Some tests failed. See details above.\n")
        
        if result.failures:
            print("Failures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print("Errors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(
        description="Run Yeast Fusion Segmenter tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_tests.py                    # Run all tests
  python tests/run_tests.py --verbose          # Verbose output
  python tests/run_tests.py --format-only      # Only format tests
  python tests/run_tests.py --integration-only # Only integration tests
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--format-only', action='store_true',
                       help='Only run format validation tests')
    parser.add_argument('--integration-only', action='store_true',
                       help='Only run integration tests')
    
    args = parser.parse_args()
    
    if args.format_only and args.integration_only:
        print("Error: Cannot specify both --format-only and --integration-only")
        sys.exit(1)
    
    verbosity = 2 if args.verbose else 1
    
    success = run_all_tests(
        verbose=verbosity,
        integration_only=args.integration_only,
        format_only=args.format_only
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
