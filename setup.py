"""
Setup script for the Adaptive Disaster Evacuation Route Optimization System.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="disaster-evacuation-routing",
    version="1.0.0",
    author="DAA Course Project",
    description="Adaptive Disaster Evacuation Route Optimization System using Dynamic Weighted Graph Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pytest>=7.0.0",
        "hypothesis>=6.0.0",
        "matplotlib>=3.5.0",
        "networkx>=2.8.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "viz": [
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
        ],
    },
)