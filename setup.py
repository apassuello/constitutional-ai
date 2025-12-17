#!/usr/bin/env python3
"""
Constitutional AI - Setup Script

A Python library for implementing Constitutional AI principles for
evaluating and training language models.
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "A Python library for implementing Constitutional AI"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = ["torch>=1.13.0", "transformers>=4.30.0", "tqdm>=4.65.0"]

setup(
    name="constitutional-ai",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python library for implementing Constitutional AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/constitutional-ai",
    packages=find_packages(exclude=["tests*", "demos*", "examples*", "docs*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
        ],
        "api": [
            "requests>=2.28.0",
        ],
        "demos": [
            "gradio>=3.0.0",
            "matplotlib>=3.5.0",
        ],
    },
    keywords=[
        "constitutional-ai",
        "ai-safety",
        "machine-learning",
        "natural-language-processing",
        "transformers",
        "language-models",
        "reinforcement-learning",
        "rlhf",
        "rlaif",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/constitutional-ai/issues",
        "Source": "https://github.com/yourusername/constitutional-ai",
        "Documentation": "https://github.com/yourusername/constitutional-ai/blob/main/docs",
    },
    include_package_data=True,
)
