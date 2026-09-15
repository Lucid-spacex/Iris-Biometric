"""Synthetic eye-image generator.

Produces fake but geometrically plausible "eye" images (dark pupil circle,
textured iris ring, lighter sclera) purely so the pipeline and the demo app
have something to run on without real iris data or CASIA access. This is
explicitly NOT biometric data and must never be described as such.
"""

from __future__ import annotations

import numpy as np


def make_synthetic_eye(
    seed: int,
    size: int = 320,
    pupil_r: int = 40,
    iris_r: int = 110,
    textured: bool = True,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cx, cy = size // 2, size // 2

    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.hypot(xx - cx, yy - cy)

    img = np.full((size, size), 200, dtype=np.float64)  # sclera background

    iris_mask = dist <= iris_r
    if textured:
        # Deterministic-per-seed pseudo texture: overlapping sinusoidal rings
        # + angular streaks, so different seeds produce different "codes".
        theta = np.arctan2(yy - cy, xx - cx)
        radial = dist / iris_r
        texture = (
            60 * np.sin(radial * 40 + rng.uniform(0, 6.28))
            + 40 * np.sin(theta * rng.integers(6, 20) + rng.uniform(0, 6.28))
            + rng.normal(0, 8, size=(size, size))
        )
        img = np.where(iris_mask, 110 + texture, img)
    else:
        img = np.where(iris_mask, 110, img)

    pupil_mask = dist <= pupil_r
    img = np.where(pupil_mask, 10, img)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def make_synthetic_rescan(base_seed: int, session: int, noise_std: float = 5.0) -> np.ndarray:
    """A second (or third, ...) "capture session" of the SAME synthetic
    identity as `make_synthetic_eye(seed=base_seed)`: identical underlying
    texture, plus small session-to-session sensor noise. Use this - not a
    different seed - to generate genuine re-scan probes; a different seed
    produces an unrelated identity, not a re-scan of the same one."""
    img = make_synthetic_eye(seed=base_seed, textured=True)
    if session <= 0:
        return img
    rng = np.random.default_rng(base_seed + session)
    noise = rng.normal(0, noise_std, size=img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def make_synthetic_gallery(n_identities: int, images_per_identity: int = 1, seed0: int = 0):
    """Yields (identity_key, image) pairs. Small variations per image
    simulate different capture sessions of the "same" synthetic identity."""
    out = []
    for i in range(n_identities):
        base_seed = seed0 + i * 1000
        for j in range(images_per_identity):
            img = make_synthetic_rescan(base_seed, session=j)
            out.append((f"S{1000+i}-L", img))
    return out
