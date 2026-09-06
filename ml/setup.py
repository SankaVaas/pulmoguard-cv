from setuptools import find_packages, setup

setup(
    name="pulmoguard-ml",
    version="1.0.0",
    description="Uncertainty-aware chest X-ray pneumonia triage: training, evaluation, and inference core.",
    packages=find_packages(include=["pulmoguard", "pulmoguard.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "pillow>=10.0.0",
    ],
)
