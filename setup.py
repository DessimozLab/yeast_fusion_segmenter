#!/usr/bin/env python3
"""
Setup script for yeast-fusion-segmenter package.

This is a compatibility setup.py for systems that don't support pyproject.toml.
The primary configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.MD'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="yeast-fusion-segmenter",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep learning-based tool for segmenting yeast cells in fusion experiments using YOLOv8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DessimozLab/yeast_fusion_segmenter",
    project_urls={
        "Bug Tracker": "https://github.com/DessimozLab/yeast_fusion_segmenter/issues",
        "Repository": "https://github.com/DessimozLab/yeast_fusion_segmenter",
        "Documentation": "https://github.com/DessimozLab/yeast_fusion_segmenter/blob/main/README.MD",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pillow>=9.5.0",
        "h5py>=3.8.0",
        "scikit-image>=0.20.0",
        "opencv-python>=4.7.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "scikit-learn>=1.2.0",
        "ultralytics>=8.0.0",
        "czifile>=2019.7.2",
        "pyimagej>=1.4.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "isort>=5.0",
        ],
        "jupyter": [
            "jupyterlab>=3.6.0",
            "jupyterlab-widgets>=3.0.0",
            "ipywidgets>=8.0.0",
            "nbconvert>=7.3.0",
            "notebook>=6.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-yolo=yeast_fusion_segmenter.train_yolo:main",
            "batch-predict=yeast_fusion_segmenter.batch_predict:main",
            "annotate-images=yeast_fusion_segmenter.annotate_images:main",
        ],
    },
    include_package_data=True,
    package_data={
        "yeast_fusion_segmenter": ["*.yaml", "*.yml", "*.pt"],
    },
    keywords=[
        "yeast", 
        "cell-segmentation", 
        "deep-learning", 
        "computer-vision", 
        "yolo", 
        "microscopy", 
        "image-processing",
        "biology",
        "fusion-experiments"
    ],
    zip_safe=False,
)