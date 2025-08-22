from setuptools import setup, find_packages

setup(
    name="hdb-resale-prediction",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0"
    ]
)