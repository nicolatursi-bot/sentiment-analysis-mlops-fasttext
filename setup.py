"""
Setup configuration for Sentiment Analysis MLOps project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sentiment-analysis-mlops",
    version="1.0.0",
    author="MLOps Innovators Inc.",
    author_email="contact@mlopsinnovators.com",
    description="MLOps Sentiment Analysis with FastText for online reputation monitoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mlopsinnovators/sentiment-analysis-mlops-fasttext",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.10",
    install_requires=[
        "fasttext==0.9.2",
        "scikit-learn==1.3.2",
        "pandas==2.1.1",
        "numpy==1.24.3",
        "datasets==2.14.5",
        "scipy==1.11.4",
    ],
    extras_require={
        "dev": [
            "pytest==7.4.3",
            "pytest-cov==4.1.0",
            "flake8==6.1.0",
            "black==23.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sentiment-mlops=src.main:main",
        ],
    },
)