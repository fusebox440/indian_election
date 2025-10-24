# Twitter Sentiment Analysis for Indian Elections
# Modern, production-ready Python package

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="twitter-sentiment-election-analysis",
    version="2.0.0",
    author="Lakshya Khetan",
    author_email="lakshyaketan00@gmail.com",
    description="A comprehensive machine learning pipeline for analyzing Twitter sentiment to predict Indian election outcomes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fusebox440/indian_election",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    python_requires=">=3.8",
    
    install_requires=requirements,
    
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
            "pre-commit>=2.15",
        ],
        "gpu": [
            "tensorflow-gpu>=2.8.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ]
    },
    
    entry_points={
        "console_scripts": [
            "twitter-sentiment=src.cli.main:main",
        ],
    },
    
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    
    include_package_data=True,
    
    keywords=[
        "sentiment analysis",
        "twitter api",
        "machine learning",
        "deep learning",
        "election prediction",
        "nlp",
        "lstm",
        "glove embeddings",
        "political analysis"
    ],
    
    project_urls={
        "Bug Reports": "https://github.com/fusebox440/indian_election/issues",
        "Source": "https://github.com/fusebox440/indian_election",
        "Documentation": "https://github.com/fusebox440/indian_election/wiki",
    },
)