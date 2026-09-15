"""Stage 3-4 - Normalisation (Daugman rubber-sheet model) and enhancement.

Produces a single canonical unwrapped strip that feeds both the Hamming
and Euclidean encoding branches (Section 3.3.1), so neither branch ever
sees a differently pre-processed image.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import DEFAULT_CONFIG
from .segmentation import Circle


@dataclass(frozen=True)
class NormalizedIris:
    strip: np.ndarray          # (angular_resolution, radial_resolution) float32, 0..1
    specular_mask: np.ndarray  # True where a specular highlight was interpolated
    validity_mask: np.ndarray  # True where the sample is usable (not off-image / not un-fixable specular)


def _rubber_sheet_coords(pupil: Circle, iris: Circle, n_r: int, n_theta: int) -> tuple[np.ndarray, np.ndarray]:
    """Map each (r, theta) sample to source-image (x, y), per Daugman's model.

    Each angular ray is referenced to its own pupil/iris boundary points so
    the mapping stays exact even when pupil and iris centres are not
    perfectly concentric.
    """
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    rs = np.linspace(0.0, 1.0, n_r, endpoint=False)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    px = pupil.x + pupil.r * cos_t
    py = pupil.y + pupil.r * sin_t
    ix = iris.x + iris.r * cos_t
    iy = iris.y + iris.r * sin_t

    # xs[theta, r] = (1-r)*pupil_boundary + r*iris_boundary
    xs = (1.0 - rs)[None, :] * px[:, None] + rs[None, :] * ix[:, None]
    ys = (1.0 - rs)[None, :] * py[:, None] + rs[None, :] * iy[:, None]
    return xs, ys


def normalize_iris(
    image: np.ndarray,
    pupil: Circle,
    iris: Circle,
    config=DEFAULT_CONFIG,
) -> NormalizedIris:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape

    n_r, n_theta = config.radial_resolution, config.angular_resolution
    xs, ys = _rubber_sheet_coords(pupil, iris, n_r, n_theta)

    on_image = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    xs_c = np.clip(xs, 0, w - 1)
    ys_c = np.clip(ys, 0, h - 1)

    strip = cv2.remap(
        image.astype(np.float32),
        xs_c.astype(np.float32),
        ys_c.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    strip = strip / 255.0

    # Specular highlight detection: near-saturated pixels.
    specular_mask = strip > 0.95

    strip_clean = strip.copy()
    if specular_mask.any():
        strip_clean = _interpolate_specular(strip_clean, specular_mask)

    # Validity: on-image AND not an unrecoverable specular region
    # (defined here as > 60% of the row saturated, which neighbour-mean
    # interpolation cannot plausibly fix).
    unfixable_rows = specular_mask.mean(axis=1) > 0.6
    validity_mask = on_image.copy()
    validity_mask[unfixable_rows, :] = False

    strip_eq = _equalize(strip_clean)

    return NormalizedIris(strip=strip_eq, specular_mask=specular_mask, validity_mask=validity_mask)


def _interpolate_specular(strip: np.ndarray, mask: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Iterative neighbour-mean interpolation over masked (specular) pixels."""
    out = strip.copy()
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    for _ in range(iterations):
        if not mask.any():
            break
        neighbour_sum = cv2.filter2D(out, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        neighbour_count = cv2.filter2D((~mask).astype(np.float32), -1, kernel, borderType=cv2.BORDER_REPLICATE)
        with np.errstate(invalid="ignore", divide="ignore"):
            filled = np.where(neighbour_count > 0, neighbour_sum / np.maximum(neighbour_count, 1e-6), out)
        out = np.where(mask, filled, out)
    return out


def _equalize(strip: np.ndarray) -> np.ndarray:
    """Global histogram equalisation (CLAHE/local equalisation considered and
    rejected for this stage, per Section 4.3.3)."""
    img8 = np.clip(strip * 255.0, 0, 255).astype(np.uint8)
    eq = cv2.equalizeHist(img8)
    return eq.astype(np.float32) / 255.0
