"""
PulmoGuard: Selective-Prediction Chest X-Ray Triage System.

This package implements an uncertainty-aware pneumonia classifier that
abstains from predicting on cases it is not confident about, rather than
forcing a diagnosis. The headline evaluation artifact is a risk-coverage
curve: model accuracy as a function of how much of the test set it is
willing to answer.
"""

__version__ = "1.0.0"
