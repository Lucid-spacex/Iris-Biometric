"""Stage 13-14 - Evaluation: ROC/EER threshold derivation and classification
metrics, per Section 3.7 (Rule Extraction Process) and Section 3.8 (Model
Performance Evaluation).

Given a set of probe images with KNOWN ground-truth identity, every probe
is compared against every enrolled (decrypted) template, and each
comparison is labelled "genuine" (same identity) or "impostor" (different
identity). From those labelled distance scores this module:

  1. Derives the operating threshold tau independently for Hamming and
     Euclidean distance, at the point where False Accept Rate and False
     Reject Rate are equal (Equal Error Rate), matching Section 3.7.
  2. Reports Accuracy, Precision, Recall, and F1 at that threshold, plus
     FAR/FRR and the raw TP/TN/FP/FN counts, matching Section 3.8's
     definitions exactly: a genuine comparison AT OR BELOW the threshold
     is a true positive; an impostor comparison at or below the threshold
     is a false positive.

IMPORTANT: this module makes no claim about what the probes represent -
that is entirely up to the caller. Pass synthetic probes and you get a
software-correctness check. Real biometric performance numbers require
this to be run against a real iris dataset (Objective i / Section 1.7).
Every result carries a `data_source` label for exactly this reason - it
should never be silently dropped when these numbers are reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import DEFAULT_CONFIG
from .matching import euclidean_distance, hamming_distance
from .pipeline import EncryptedGallery, PipelineStageFailure, process_image
from .template import build_template


@dataclass
class ScoreSet:
    hamming_genuine: list = field(default_factory=list)
    hamming_impostor: list = field(default_factory=list)
    euclidean_genuine: list = field(default_factory=list)
    euclidean_impostor: list = field(default_factory=list)


def collect_scores(
    gallery: EncryptedGallery,
    probes: list[tuple[str, np.ndarray]],
    config=DEFAULT_CONFIG,
) -> tuple[ScoreSet, list[str]]:
    """probes: (true_identity_key, image) pairs. Compares each probe against
    every enrolled record, labelling each of the resulting comparisons
    genuine or impostor by whether the identities match.

    Returns the labelled scores plus a list of probe identities skipped
    because segmentation failed, so a batch evaluation run never silently
    drops probes from the reported counts.
    """
    enrolled = gallery.decrypt_all()
    scores = ScoreSet()
    skipped: list[str] = []

    for true_identity, image in probes:
        try:
            processed = process_image(image, true_identity, config=config)
        except PipelineStageFailure:
            skipped.append(true_identity)
            continue

        probe_record = build_template(
            true_identity, processed.hamming_feature, gallery.projector, processed.raw_euclidean_vector
        )
        for enrolled_record in enrolled:
            hd = hamming_distance(probe_record, enrolled_record)
            ed = euclidean_distance(probe_record, enrolled_record)
            is_genuine = true_identity == enrolled_record.identity_key
            if is_genuine:
                scores.hamming_genuine.append(hd)
                scores.euclidean_genuine.append(ed)
            else:
                scores.hamming_impostor.append(hd)
                scores.euclidean_impostor.append(ed)

    return scores, skipped


def _far_frr_at(genuine: np.ndarray, impostor: np.ndarray, threshold: float) -> tuple[float, float]:
    frr = float(np.mean(genuine > threshold)) if len(genuine) else 0.0
    far = float(np.mean(impostor <= threshold)) if len(impostor) else 0.0
    return far, frr


def roc_points(genuine: list[float], impostor: list[float]) -> list[tuple[float, float, float]]:
    """Sweeps every observed distance value as a candidate threshold,
    returning (threshold, FAR, FRR) triples - the ROC curve of Section 3.7."""
    genuine_arr = np.asarray(genuine)
    impostor_arr = np.asarray(impostor)
    candidates = sorted(set(list(genuine) + list(impostor)))
    return [(t, *_far_frr_at(genuine_arr, impostor_arr, t)) for t in candidates]


class InsufficientScoresError(ValueError):
    """Raised when there are no genuine or no impostor scores to compare -
    EER is undefined without both."""


def find_eer(genuine: list[float], impostor: list[float]) -> tuple[float, float]:
    """Returns (threshold, eer) at the point on the ROC curve where FAR and
    FRR are closest (Equal Error Rate), per Section 3.7."""
    if not genuine or not impostor:
        raise InsufficientScoresError(
            "Need at least one genuine and one impostor score to derive an EER threshold."
        )
    points = roc_points(genuine, impostor)
    threshold, far, frr = min(points, key=lambda p: abs(p[1] - p[2]))
    return threshold, (far + frr) / 2.0


@dataclass(frozen=True)
class MetricReport:
    """The four Section 3.8 classification metrics plus FAR/FRR, at a
    specific threshold, for one distance metric (Hamming or Euclidean)."""

    threshold: float
    eer: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    far: float
    frr: float
    tp: int
    tn: int
    fp: int
    fn: int
    n_genuine: int
    n_impostor: int


def classification_metrics(genuine: list[float], impostor: list[float], threshold: float) -> MetricReport:
    """Accuracy/Precision/Recall/F1 (Section 3.8), applying the definition
    given there verbatim: a genuine comparison at or below the threshold is
    a true positive; an impostor comparison at or below the threshold is a
    false positive."""
    genuine_arr = np.asarray(genuine)
    impostor_arr = np.asarray(impostor)

    tp = int(np.sum(genuine_arr <= threshold))
    fn = int(np.sum(genuine_arr > threshold))
    tn = int(np.sum(impostor_arr > threshold))
    fp = int(np.sum(impostor_arr <= threshold))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr = fn / (fn + tp) if (fn + tp) else 0.0

    return MetricReport(
        threshold=threshold,
        eer=(far + frr) / 2.0,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        far=far,
        frr=frr,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        n_genuine=len(genuine),
        n_impostor=len(impostor),
    )


@dataclass(frozen=True)
class EvaluationResult:
    hamming: MetricReport
    euclidean: MetricReport
    skipped_probes: list[str]
    data_source: str  # e.g. "synthetic (software-correctness check)" or a real dataset name


def evaluate(
    gallery: EncryptedGallery,
    probes: list[tuple[str, np.ndarray]],
    data_source: str,
    config=DEFAULT_CONFIG,
) -> EvaluationResult:
    """Runs Stages 13-14 end to end: score collection, EER threshold
    derivation, and classification metrics, for both distance metrics.

    `data_source` is mandatory and always carried on the result so a
    synthetic-data run can never be mistaken for a real-dataset result
    downstream (e.g. in a report table or a UI panel).
    """
    scores, skipped = collect_scores(gallery, probes, config=config)

    h_threshold, _ = find_eer(scores.hamming_genuine, scores.hamming_impostor)
    e_threshold, _ = find_eer(scores.euclidean_genuine, scores.euclidean_impostor)

    hamming_report = classification_metrics(scores.hamming_genuine, scores.hamming_impostor, h_threshold)
    euclidean_report = classification_metrics(scores.euclidean_genuine, scores.euclidean_impostor, e_threshold)

    return EvaluationResult(
        hamming=hamming_report,
        euclidean=euclidean_report,
        skipped_probes=skipped,
        data_source=data_source,
    )
