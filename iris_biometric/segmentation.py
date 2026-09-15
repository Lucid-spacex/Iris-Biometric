"""Stage 2 - Iris segmentation (boundary localisation only).

Finds the pupil and iris-limbus boundaries as circles. Deliberately does
NOT classify eyelid/eyelash occlusion here - that mask is attached later
(see features.py) so segmentation itself never needs to be redesigned to
add it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    r: float


@dataclass(frozen=True)
class SegmentationResult:
    ok: bool
    pupil: Circle | None = None
    iris: Circle | None = None
    reason: str = ""


class SegmentationInputError(ValueError):
    """Raised for malformed input (not a normal failed-to-segment case)."""


def _validate_image(image: np.ndarray) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise SegmentationInputError("Input image is empty or not a valid array")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim != 2:
        raise SegmentationInputError(f"Unsupported image shape: {image.shape}")
    return image


def _find_circles(gray: np.ndarray, min_r: int, max_r: int, param2: float) -> np.ndarray | None:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=gray.shape[0] / 4,
        param1=100,   # upper Canny threshold used internally by HoughCircles
        param2=param2,
        minRadius=min_r,
        maxRadius=max_r,
    )
    return circles


def segment_iris(
    image: np.ndarray,
    pupil_radius_band: tuple[int, int] | None = None,
    iris_radius_band: tuple[int, int] | None = None,
) -> SegmentationResult:
    """Locate the pupil and iris boundaries.

    Supports both fixed and adaptive radius bands so real phone uploads,
    different sensor resolutions, and synthetic images segment reliably.
    """
    gray = _validate_image(image)
    h, w = gray.shape

    # Calculate adaptive radius bands if not explicitly provided
    if pupil_radius_band is None:
        pupil_radius_band = (max(8, int(round(h * 0.04))), max(30, int(round(h * 0.25))))
    if iris_radius_band is None:
        iris_radius_band = (max(40, int(round(h * 0.20))), max(100, int(round(h * 0.65))))

    # Multi-threshold pupil search (robust against varied contrast/lighting)
    pupil_circles = None
    for p2 in (30, 25, 20):
        pupil_circles = _find_circles(gray, *pupil_radius_band, param2=p2)
        if pupil_circles is not None:
            break

    if pupil_circles is None:
        return SegmentationResult(ok=False, reason="no_pupil_boundary_found")
    px, py, pr = pupil_circles[0][0]
    pupil = Circle(float(px), float(py), float(pr))

    # Multi-threshold iris search
    iris_circles = None
    for p2 in (40, 30, 25):
        iris_circles = _find_circles(gray, *iris_radius_band, param2=p2)
        if iris_circles is not None:
            break

    if iris_circles is None:
        return SegmentationResult(ok=False, reason="no_iris_boundary_found")

    # Prefer the iris candidate most concentric with the detected pupil.
    candidates = iris_circles[0]
    dists = np.hypot(candidates[:, 0] - px, candidates[:, 1] - py)
    ix, iy, ir = candidates[int(np.argmin(dists))]
    iris = Circle(float(ix), float(iy), float(ir))

    center_offset = float(np.hypot(iris.x - pupil.x, iris.y - pupil.y))
    if center_offset > 0.35 * iris.r:
        return SegmentationResult(ok=False, reason="pupil_iris_not_concentric")

    ratio = iris.r / pupil.r if pupil.r > 0 else 0.0
    if not (1.15 <= ratio <= 8.0):
        return SegmentationResult(ok=False, reason="implausible_iris_pupil_ratio")

    return SegmentationResult(ok=True, pupil=pupil, iris=iris)
