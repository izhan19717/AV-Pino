"""
Setup script for AV-PINO Motor Fault Diagnosis System
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="av-pino-motor-fault-diagnosis",
    version="0.1.0",
    author="AV-PINO Research Team",
    author_email="research@av-pino.org",
    description="Adaptive Variational Physics-Informed Neural Operator for Motor Fault Diagnosis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/av-pino/motor-fault-diagnosis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "gpu": [
            "cupy-cuda11x>=11.0.0",
        ],
        "edge": [
            "onnx>=1.12.0",
            "onnxruntime>=1.12.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "cupy-cuda11x>=11.0.0",
            "onnx>=1.12.0",
            "onnxruntime>=1.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "av-pino-train=training.training_engine:main",
            "av-pino-infer=inference.realtime_inference:main",
            "av-pino-validate=validation.benchmarking_suite:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
)