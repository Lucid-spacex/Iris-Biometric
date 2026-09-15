"""Stage 5 - Feature encoding.

A single 1D Log-Gabor filtering pass along each angular row produces both
branch representations from one filter application (Section 4.4):
  - Hamming branch: 2-bit phase-quadrant IrisCode (sign of real, sign of imaginary)
  - Euclidean branch: the unquantised complex response, kept for Stage 6

A noise mask is derived from amplitude: samples whose filter response is
too weak to be a reliable phase estimate are flagged unusable, combined
downstream (Stage 6) with the pre-processing validity mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_CONFIG
from .normalization import NormalizedIris


@dataclass(frozen=True)
class EncodedIris:
    iris_code: np.ndarray       # bool, shape (n_theta, n_r, 2) -> flattened to n_theta*n_r*2 bits
    complex_response: np.ndarray  # complex64, shape (n_theta, n_r) - unquantised, for Euclidean branch
    noise_mask: np.ndarray      # bool, shape (n_theta, n_r); True = reliable / usable


def _log_gabor_1d(n: int, wavelength: float, sigma_on_f: float) -> np.ndarray:
    """1D Log-Gabor filter in the frequency domain, DC forced to zero, only
    the positive-frequency (analytic) response retained."""
    freqs = np.fft.fftfreq(n)
    f0 = 1.0 / wavelength
    filt = np.zeros(n, dtype=np.float64)
    positive = freqs > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        filt[positive] = np.exp(
            -(np.log(freqs[positive] / f0) ** 2) / (2 * np.log(sigma_on_f) ** 2)
        )
    filt[0] = 0.0  # force DC to zero
    return filt


def encode_iris(normalized: NormalizedIris, config=DEFAULT_CONFIG) -> EncodedIris:
    strip = normalized.strip  # shape (n_theta, n_r)
    n_theta, n_r = strip.shape

    log_gabor = _log_gabor_1d(n_r, config.log_gabor_wavelength, config.log_gabor_sigma_ratio)

    # Filter each angular row independently along the radial axis.
    row_fft = np.fft.fft(strip.astype(np.float64), axis=1)
    filtered_fft = row_fft * log_gabor[None, :]
    complex_response = np.fft.ifft(filtered_fft, axis=1).astype(np.complex64)

    # 2-bit phase-quadrant quantisation: sign of real part, sign of imaginary part.
    bit_real = complex_response.real >= 0
    bit_imag = complex_response.imag >= 0
    iris_code = np.stack([bit_real, bit_imag], axis=-1)  # (n_theta, n_r, 2)

    amplitude = np.abs(complex_response)
    amplitude_ok = amplitude >= config.min_response_amplitude
    noise_mask = amplitude_ok & normalized.validity_mask

    return EncodedIris(iris_code=iris_code, complex_response=complex_response, noise_mask=noise_mask)
