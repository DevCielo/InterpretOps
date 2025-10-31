from setuptools import setup, find_packages

setup(
    name="interpretops",
    version="0.1.0",
    description="Interpretation operations toolkit for mechanistic interpretability",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "h5py>=3.9.0",
        "numpy>=1.24.0",
        "click>=8.1.0",
    ],
    entry_points={
        "console_scripts": [
            "mechctl=interpretops.cli:main",
        ],
    },
    python_requires=">=3.8",
)

