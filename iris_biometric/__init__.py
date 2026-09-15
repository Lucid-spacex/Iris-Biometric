"""Iris biometric identification system.

Implements the pipeline described in the accompanying seminar report:
segmentation -> normalisation -> Log-Gabor encoding -> feature selection
(standardisation + PCA) -> AES-encrypted template storage -> Hamming /
Euclidean one-to-many matching -> ROC/EER-based evaluation.
"""

__version__ = "0.1.0"
