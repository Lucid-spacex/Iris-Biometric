"""Visualisation helpers for the demo web app only - not part of the
biometric pipeline itself."""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image


def _array_to_data_uri(arr: np.ndarray) -> str:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_segmentation_overlay(image: np.ndarray, seg) -> str:
    """Draw the detected pupil/iris circles on the source image."""
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis = image.copy()
    if seg.ok:
        cv2.circle(vis, (int(seg.pupil.x), int(seg.pupil.y)), int(seg.pupil.r), (255, 140, 0), 2)
        cv2.circle(vis, (int(seg.iris.x), int(seg.iris.y)), int(seg.iris.r), (80, 200, 255), 2)
    return _array_to_data_uri(vis)


def render_strip(strip: np.ndarray) -> str:
    """The unwrapped Daugman rubber-sheet strip, upscaled for visibility."""
    img8 = np.clip(strip * 255.0, 0, 255).astype(np.uint8)
    # strip shape is (n_theta, n_r); transpose so theta runs horizontally
    img8 = img8.T
    upscaled = cv2.resize(img8, (img8.shape[1] * 3, img8.shape[0] * 6), interpolation=cv2.INTER_NEAREST)
    return _array_to_data_uri(upscaled)


def render_iris_code(bits: np.ndarray) -> str:
    """The 2-bit-per-sample IrisCode as a black/white bitmap."""
    n_theta, n_r, _ = bits.shape
    flat = bits.reshape(n_theta, n_r * 2).astype(np.uint8) * 255
    img8 = flat.T  # bits run vertically, angle horizontally
    upscaled = cv2.resize(img8, (img8.shape[1] * 3, img8.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
    return _array_to_data_uri(upscaled)


def render_raw(image: np.ndarray) -> str:
    return _array_to_data_uri(image)
