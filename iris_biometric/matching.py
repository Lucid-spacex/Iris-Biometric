"""Stage 11-12 - Distance metrics and one-to-many identification.

Implements Algorithm 2 (Section 3.4): a probe template is compared against
every enrolled (decrypted) template using fractional Hamming distance over
unmasked bits, and Euclidean distance over PCA-reduced vectors. Each
comparison yields a numeric score; the threshold rule of Section 3.7
converts scores into match / non-match decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_CONFIG
from .template import TemplateRecord


def hamming_distance(probe: TemplateRecord, enrolled: TemplateRecord) -> float:
    """Fractional Hamming distance over bits unmasked in BOTH templates.

    Returns a value in [0, 1]; 0 = identical, 0.5 = chance-level for random
    binary codes. Returns 1.0 (maximal distance) if no bits are usable in
    common, so a probe with no comparable bits can never spuriously "match".
    """
    usable = (~probe.hamming_exclusion) & (~enrolled.hamming_exclusion)
    n_usable = int(usable.sum())
    if n_usable == 0:
        return 1.0
    disagreements = np.logical_xor(probe.hamming_bits[usable], enrolled.hamming_bits[usable])
    return float(disagreements.sum() / n_usable)


def euclidean_distance(probe: TemplateRecord, enrolled: TemplateRecord) -> float:
    if probe.euclidean_vector.shape != enrolled.euclidean_vector.shape:
        # Projector/template dimension mismatch — treat as maximum distance (non-match).
        return float("inf")
    diff = probe.euclidean_vector - enrolled.euclidean_vector
    return float(np.linalg.norm(diff))


@dataclass(frozen=True)
class MatchCandidate:
    identity_key: str
    hamming_distance: float
    euclidean_distance: float
    hamming_match: bool
    euclidean_match: bool


@dataclass(frozen=True)
class IdentificationResult:
    candidates: list[MatchCandidate]
    best_hamming: MatchCandidate | None
    best_euclidean: MatchCandidate | None
    hamming_threshold: float
    euclidean_threshold: float

    @property
    def hamming_decision(self) -> str:
        return self.best_hamming.identity_key if self.best_hamming else "non-match"

    @property
    def euclidean_decision(self) -> str:
        return self.best_euclidean.identity_key if self.best_euclidean else "non-match"


def identify(
    probe: TemplateRecord,
    gallery: list[TemplateRecord],
    hamming_threshold: float = DEFAULT_CONFIG.provisional_hamming_threshold,
    euclidean_threshold: float = DEFAULT_CONFIG.provisional_euclidean_threshold,
) -> IdentificationResult:
    """One-to-many identification (Algorithm 2).

    Scores the probe against every gallery record with both metrics, applies
    the threshold rule independently to each, and returns the qualifying
    candidate with the smallest distance per metric (or None -> "non-match").

    NOTE: both thresholds are the *provisional* placeholders from config.py,
    not the ROC/EER-derived values Section 3.7 specifies - those require
    real genuine/impostor distance distributions from CASIA data, which is
    not yet available. Treat match/non-match decisions from this function
    as a software-correctness demo, not a biometric performance claim.
    """
    candidates: list[MatchCandidate] = []
    for enrolled in gallery:
        hd = hamming_distance(probe, enrolled)
        ed = euclidean_distance(probe, enrolled)
        candidates.append(
            MatchCandidate(
                identity_key=enrolled.identity_key,
                hamming_distance=hd,
                euclidean_distance=ed,
                hamming_match=hd <= hamming_threshold,
                euclidean_match=ed <= euclidean_threshold,
            )
        )

    qualifying_h = [c for c in candidates if c.hamming_match]
    best_h = min(qualifying_h, key=lambda c: c.hamming_distance) if qualifying_h else None

    qualifying_e = [c for c in candidates if c.euclidean_match]
    best_e = min(qualifying_e, key=lambda c: c.euclidean_distance) if qualifying_e else None

    return IdentificationResult(
        candidates=candidates, 
        best_hamming=best_h, 
        best_euclidean=best_e,
        hamming_threshold=hamming_threshold,
        euclidean_threshold=euclidean_threshold
    )
