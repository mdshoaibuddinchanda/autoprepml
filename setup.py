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
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "jinja2>=3.0.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "ruff>=0.0.250",
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
