"""Setup configuration for AutoPrepML"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="autoprepml",
    version="1.0.0",
    author="MD Shoaibuddin Chanda",
    author_email="mdshoaibuddinchanda@gmail.com",
    description="AI-Assisted Multi-Modal Data Preprocessing Pipeline for ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mdshoaibuddinchanda/autoprepml",
    packages=find_packages(exclude=["tests*", "docs*", "examples*", "scripts*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas==2.2.2",
        "numpy==1.26.4",
        "scikit-learn==1.7.2",
        "matplotlib==3.10.7",
        "seaborn==0.13.2",
        "jinja2==3.1.6",
        "pyyaml==6.0.3",
    ],
    extras_require={
        "dev": [
            "pytest==8.4.2",
            "pytest-cov==7.0.0",
            "black==24.10.0",
            "ruff==0.8.4",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "autoprepml=autoprepml.cli:main",
        ],
    },
    project_urls={
        "Bug Tracker": "https://github.com/mdshoaibuddinchanda/autoprepml/issues",
        "Documentation": "https://github.com/mdshoaibuddinchanda/autoprepml#readme",
        "Source Code": "https://github.com/mdshoaibuddinchanda/autoprepml",
    },
)
