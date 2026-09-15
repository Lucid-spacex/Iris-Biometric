"""Central, explicit configuration for the pipeline.

Kept as one module so every stage reads the same numbers, and so the
provisional parameters called out in Chapter Four (resolution, filter
scale, PCA component count) live in one place instead of being
scattered as magic numbers through the codebase.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    # --- Normalisation (Daugman rubber-sheet unwrap) ---
    radial_resolution: int = 20       # samples along r
    angular_resolution: int = 240     # samples along theta

    # --- Log-Gabor encoding ---
    log_gabor_wavelength: float = 18.0   # centre wavelength, pixels
    log_gabor_sigma_ratio: float = 0.5   # bandwidth parameter (sigma/f0)
    min_response_amplitude: float = 1e-3  # below this -> masked as unreliable

    # --- Feature selection ---
    pca_n_components: int = 128
    euclidean_feature_source: str = "concat"  # "concat" | "amplitude" | "real"

    # --- AES ---
    aes_key_bits: int = 256  # 128 | 192 | 256, per FIPS 197

    # --- Matching / decision ---
    # Provisional until real ROC/EER derivation against CASIA data (Sec 3.7).
    provisional_hamming_threshold: float = 0.32
    provisional_euclidean_threshold: float = 12.0

    # --- Data ---
    supported_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp")


DEFAULT_CONFIG = PipelineConfig()
