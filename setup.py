from setuptools import find_packages, setup

setup(
    name="pulmoguard",
    version="1.0.0",
    description="Uncertainty-aware chest X-ray pneumonia triage with calibrated abstention.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
)
