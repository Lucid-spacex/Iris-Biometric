"""Stage 6 - Feature selection, standardisation, and PCA.

Two branches from one encoding:
  - Euclidean: flatten the complex response -> standardise -> PCA.
    Fitting (mean/scale/components) happens ONLY on a gallery set, never
    on a single probe, so no probe information can leak into the
    statistics used to score it (Section 4.5.5). This is the single most
    consequential correctness property for a fair Hamming-vs-Euclidean
    comparison, so it is enforced at the API level, not just by convention.
  - Hamming: package the binary IrisCode with a combined exclusion mask
    (encoding noise mask + optional eyelid/eyelash occlusion mask). No
    distance or matching logic lives here - see matching.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_CONFIG
from .encoding import EncodedIris


# ---------------------------------------------------------------- Hamming --

@dataclass(frozen=True)
class HammingFeature:
    bits: np.ndarray          # bool, flattened iris code bits
    exclusion_mask: np.ndarray  # bool, True = excluded (unreliable / occluded)

    @property
    def usable_bits(self) -> int:
        return int((~self.exclusion_mask).sum())

    @property
    def total_bits(self) -> int:
        return int(self.exclusion_mask.size)

    @property
    def masked_fraction(self) -> float:
        return float(self.exclusion_mask.mean())


def select_hamming_feature(encoded: EncodedIris, occlusion_mask: np.ndarray | None = None) -> HammingFeature:
    bits = encoded.iris_code.reshape(-1)
    noise_excluded = ~encoded.noise_mask.reshape(-1)
    # noise_mask is per (theta, r); iris_code has 2 bits per (theta, r) position.
    noise_excluded_bits = np.repeat(noise_excluded, 2)

    if occlusion_mask is not None:
        occlusion_excluded_bits = np.repeat(occlusion_mask.reshape(-1).astype(bool), 2)
        exclusion_mask = noise_excluded_bits | occlusion_excluded_bits
    else:
        exclusion_mask = noise_excluded_bits

    return HammingFeature(bits=bits, exclusion_mask=exclusion_mask)


# --------------------------------------------------------------- Euclidean --

def _raw_euclidean_vector(encoded: EncodedIris, source: str) -> np.ndarray:
    resp = encoded.complex_response.reshape(-1)
    if source == "concat":
        return np.concatenate([resp.real, resp.imag]).astype(np.float64)
    if source == "amplitude":
        return np.abs(resp).astype(np.float64)
    if source == "real":
        return resp.real.astype(np.float64)
    raise ValueError(f"Unknown euclidean_feature_source: {source!r}")


class EuclideanFeatureFailure(ValueError):
    pass


@dataclass
class EuclideanProjector:
    """A fitted (gallery-only) standardisation + PCA transform.

    Construct via `EuclideanProjector.fit(gallery_vectors, ...)`. `transform`
    is a pure, non-mutating function: it never updates the fitted statistics,
    so probe data cannot influence them.
    """

    scaler: StandardScaler
    pca: PCA
    n_components_effective: int
    n_components_requested: int
    warning: str | None

    @classmethod
    def fit(
        cls,
        gallery_vectors: np.ndarray,
        n_components: int = DEFAULT_CONFIG.pca_n_components,
    ) -> "EuclideanProjector":
        gallery_vectors = np.asarray(gallery_vectors, dtype=np.float64)
        if gallery_vectors.ndim != 2 or gallery_vectors.shape[0] < 2:
            raise EuclideanFeatureFailure(
                "fit() requires a gallery of 2+ samples, shape (n_samples, n_features); "
                "a bare single sample is rejected to prevent probe-derived fitting."
            )

        scaler = StandardScaler()
        scaler.fit(gallery_vectors)
        # Guard against divide-by-zero for zero-variance features: fix scale
        # to 1 so standardised value is defined as 0, not NaN/Inf.
        zero_var = scaler.scale_ == 0
        if np.any(zero_var):
            scaler.scale_ = scaler.scale_.copy()
            scaler.scale_[zero_var] = 1.0

        standardized = scaler.transform(gallery_vectors)

        n_samples, n_features = standardized.shape
        effective = min(n_components, n_samples, n_features)
        warning = None
        if effective < n_components:
            warning = (
                f"Requested {n_components} PCA components but gallery only supports "
                f"{effective} (n_samples={n_samples}, n_features={n_features}); clamped."
            )

        pca = PCA(n_components=effective, svd_solver="full")
        pca.fit(standardized)

        return cls(
            scaler=scaler,
            pca=pca,
            n_components_effective=effective,
            n_components_requested=n_components,
            warning=warning,
        )

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Pure transform - does not mutate fitted state."""
        vectors = np.asarray(vectors, dtype=np.float64)
        single = vectors.ndim == 1
        if single:
            vectors = vectors[None, :]
        standardized = self.scaler.transform(vectors)
        projected = self.pca.transform(standardized)
        return projected[0] if single else projected

    def explained_variance_ratio_sum(self) -> float:
        return float(np.sum(self.pca.explained_variance_ratio_))


def euclidean_raw_vectors(
    encoded_list: list[EncodedIris],
    source: str = DEFAULT_CONFIG.euclidean_feature_source,
) -> np.ndarray:
    """Build the raw (pre-standardisation) feature matrix for a set of encodings."""
    return np.stack([_raw_euclidean_vector(e, source) for e in encoded_list])
