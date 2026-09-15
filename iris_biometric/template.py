"""Stage 9 - Template construction.

Bundles the Hamming feature (masked bits) and the Euclidean feature
(PCA-projected vector) for one identity into a single serialisable record,
ready to be handed to crypto.py for AES encryption (Stage 10).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .features import EuclideanProjector, HammingFeature


@dataclass(frozen=True)
class TemplateRecord:
    identity_key: str
    hamming_bits: np.ndarray        # bool array
    hamming_exclusion: np.ndarray   # bool array, True = excluded
    euclidean_vector: np.ndarray    # float64, PCA-projected

    def to_bytes(self) -> bytes:
        """Serialise to a compact JSON payload (then AES-encrypted by crypto.py)."""
        payload = {
            "identity_key": self.identity_key,
            "hamming_bits": self.hamming_bits.astype(np.uint8).tolist(),
            "hamming_exclusion": self.hamming_exclusion.astype(np.uint8).tolist(),
            "euclidean_vector": self.euclidean_vector.tolist(),
        }
        return json.dumps(payload).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "TemplateRecord":
        payload = json.loads(data.decode("utf-8"))
        return cls(
            identity_key=payload["identity_key"],
            hamming_bits=np.array(payload["hamming_bits"], dtype=bool),
            hamming_exclusion=np.array(payload["hamming_exclusion"], dtype=bool),
            euclidean_vector=np.array(payload["euclidean_vector"], dtype=np.float64),
        )


def build_template(
    identity_key: str,
    hamming_feature: HammingFeature,
    euclidean_projector: EuclideanProjector,
    euclidean_raw_vector: np.ndarray,
) -> TemplateRecord:
    projected = euclidean_projector.transform(euclidean_raw_vector)
    return TemplateRecord(
        identity_key=identity_key,
        hamming_bits=hamming_feature.bits,
        hamming_exclusion=hamming_feature.exclusion_mask,
        euclidean_vector=np.asarray(projected, dtype=np.float64),
    )
