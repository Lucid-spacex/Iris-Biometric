"""MMU1 Iris Dataset evaluation runner.

Loads real iris images from the MMU1 dataset (Multimedia University),
enrolls a gallery, and runs the evaluation pipeline (ROC/EER derivation,
Hamming vs Euclidean matching, Accuracy, Precision, Recall, F1, FAR, FRR).

All outputs explicitly state that results are measured on MMU1 as an interim
substitute benchmark pending CASIA-Iris-Thousand access.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from iris_biometric.config import DEFAULT_CONFIG
from iris_biometric.evaluation import EvaluationResult, evaluate
from iris_biometric.identity import (
    DataSource,
    IrisRecord,
    dataset_statistics,
    discover_dataset,
)
from iris_biometric.pipeline import (
    EncryptedGallery,
    enroll_gallery,
)

MMU1_LABEL = "MMU1 Iris Database (Interim substitute dataset pending CASIA-Iris-Thousand access)"


def load_grayscale_image(path: Path) -> np.ndarray:
    """Read an image file as 8-bit grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image at {path}")
    return img


def find_default_mmu_dir() -> Path | None:
    """Check common local paths where MMU1 might be placed."""
    candidates = [
        Path("data/mmu1"),
        Path("data/MMU-Iris-Database"),
        Path("data/mmu-iris-dataset"),
        Path("MMU-Iris-Database"),
        Path("../data/mmu1"),
        Path.home() / "Downloads" / "MMU-Iris-Database",
        Path.home() / "Downloads" / "mmu-iris-dataset",
        Path.home() / "Downloads" / "mmu1",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None


def partition_dataset(
    records: List[IrisRecord],
) -> Tuple[List[IrisRecord], List[IrisRecord]]:
    """Partition records by eye identity (S<subject>-<eye>).

    The first available image for each identity is assigned to the enrolment
    gallery. All subsequent images for that identity become evaluation probes.
    """
    by_identity: Dict[str, List[IrisRecord]] = {}
    for r in records:
        by_identity.setdefault(r.identity_key, []).append(r)

    gallery_records: List[IrisRecord] = []
    probe_records: List[IrisRecord] = []

    for identity_key in sorted(by_identity.keys()):
        # Sort by image index so enrolment is consistent
        imgs = sorted(by_identity[identity_key], key=lambda x: str(x.index))
        gallery_records.append(imgs[0])
        probe_records.extend(imgs[1:])

    return gallery_records, probe_records


def run_mmu_evaluation(data_dir: Path, output_json: Path | None = None) -> EvaluationResult:
    print("=" * 72)
    print("IRIS BIOMETRIC IDENTIFICATION - MMU1 BENCHMARK EVALUATION")
    print("=" * 72)
    print(f"Data directory: {data_dir.resolve()}")
    print(f"Data source   : {MMU1_LABEL}")
    print("-" * 72)

    # 1. Discover records
    records = discover_dataset(data_dir, source=DataSource.MMU1_STUDY_DATA)
    if not records:
        print(f"Error: No valid iris images found under {data_dir}")
        print("Expected structure: <subject>/<left|right>/<image>.bmp")
        sys.exit(1)

    stats = dataset_statistics(records)
    print(f"Discovered images      : {stats['n_images']}")
    print(f"Unique subjects        : {stats['n_subjects']}")
    print(f"Distinct eye identities: {stats['n_identities']}")
    print(f"Images per eye identity: {stats['min_images_per_identity']} to {stats['max_images_per_identity']}")
    print("-" * 72)

    # 2. Partition into Gallery (1 per eye) and Probes (remaining)
    gallery_recs, probe_recs = partition_dataset(records)
    print(f"Partition: {len(gallery_recs)} images for enrolment gallery (1 per eye)")
    print(f"           {len(probe_recs)} images for evaluation probe set")
    print("-" * 72)

    # 3. Load enrolment images and enroll gallery
    print("Enrolling gallery and fitting PCA Euclidean projector...")
    gallery_images: List[Tuple[str, np.ndarray]] = []
    for r in gallery_recs:
        try:
            img = load_grayscale_image(r.path)
            gallery_images.append((r.identity_key, img))
        except Exception as e:
            print(f"Warning: could not read {r.path}: {e}")

    gallery, skipped_gallery = enroll_gallery(gallery_images)
    print(f"Gallery enrolled: {len(gallery.records)} templates encrypted with AES-256")
    print(f"PCA components fitted: {gallery.projector.n_components_effective}")
    if skipped_gallery:
        print(f"Skipped gallery images (segmentation failed): {len(skipped_gallery)}")

    # 4. Load probe images
    print("Loading evaluation probes...")
    probe_images: List[Tuple[str, np.ndarray]] = []
    for r in probe_recs:
        try:
            img = load_grayscale_image(r.path)
            probe_images.append((r.identity_key, img))
        except Exception as e:
            print(f"Warning: could not read {r.path}: {e}")

    # 5. Run full biometric evaluation (Stages 13-14)
    print(f"Running evaluation against {len(probe_images)} probes...")
    result = evaluate(
        gallery=gallery,
        probes=probe_images,
        data_source=MMU1_LABEL,
        config=DEFAULT_CONFIG,
    )

    # 6. Format and display results
    print("\n" + "=" * 72)
    print("FINAL EVALUATION METRICS (Equal Error Rate operating point)")
    print("=" * 72)
    print(f"DATASET: {result.data_source}\n")

    header = f"{'Metric':<11} | {'EER Thresh':<10} | {'EER':<7} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<6} | {'FAR':<7} | {'FRR':<7}"
    print(header)
    print("-" * len(header))

    h = result.hamming
    print(
        f"{'Hamming':<11} | {h.threshold:<10.4f} | {h.eer*100:<6.2f}% | "
        f"{h.accuracy*100:<7.2f}% | {h.precision*100:<8.2f}% | {h.recall*100:<7.2f}% | "
        f"{h.f1:<6.4f} | {h.far*100:<6.2f}% | {h.frr*100:<6.2f}%"
    )

    e = result.euclidean
    print(
        f"{'Euclidean':<11} | {e.threshold:<10.2f} | {e.eer*100:<6.2f}% | "
        f"{e.accuracy*100:<7.2f}% | {e.precision*100:<8.2f}% | {e.recall*100:<7.2f}% | "
        f"{e.f1:<6.4f} | {e.far*100:<6.2f}% | {e.frr*100:<6.2f}%"
    )
    print("-" * len(header))

    print("\nDetailed Comparison Counts:")
    print(f"  Hamming   : TP={h.tp}, TN={h.tn}, FP={h.fp}, FN={h.fn} (Genuine={h.n_genuine}, Impostor={h.n_impostor})")
    print(f"  Euclidean : TP={e.tp}, TN={e.tn}, FP={e.fp}, FN={e.fn} (Genuine={e.n_genuine}, Impostor={e.n_impostor})")
    if result.skipped_probes:
        print(f"  Skipped probe images (failed segmentation): {len(result.skipped_probes)} / {len(probe_images)}")
    print("=" * 72)

    # 8. Direct comparison and conclusion
    print("\n" + "=" * 72)
    print("HAMMING VS EUCLIDEAN COMPARISON")
    print("=" * 72)

    # Compare based on accuracy (primary metric for overall performance)
    if h.accuracy > e.accuracy:
        winner = "Hamming"
        accuracy_diff = (h.accuracy - e.accuracy) * 100
        conclusion = f"Hamming distance outperformed Euclidean distance on this evaluation set: {h.accuracy*100:.2f}% accuracy vs {e.accuracy*100:.2f}% accuracy (+{accuracy_diff:.2f} percentage points)"
    elif e.accuracy > h.accuracy:
        winner = "Euclidean"
        accuracy_diff = (e.accuracy - h.accuracy) * 100
        conclusion = f"Euclidean distance outperformed Hamming distance on this evaluation set: {e.accuracy*100:.2f}% accuracy vs {h.accuracy*100:.2f}% accuracy (+{accuracy_diff:.2f} percentage points)"
    else:
        winner = "Tie"
        conclusion = f"Hamming and Euclidean distances achieved identical accuracy on this evaluation set: {h.accuracy*100:.2f}%"

    print(f"\nPRIMARY COMPARISON (Accuracy):")
    print(f"  Winner: {winner}")
    print(f"  {conclusion}")

    # Secondary comparisons
    print(f"\nSECONDARY METRICS:")
    print(f"  EER (lower is better): Hamming {h.eer*100:.2f}% vs Euclidean {e.eer*100:.2f}%")
    print(f"  F1 Score (higher is better): Hamming {h.f1:.4f} vs Euclidean {e.f1:.4f}")
    print(f"  Precision (higher is better): Hamming {h.precision*100:.2f}% vs Euclidean {e.precision*100:.2f}%")
    print(f"  Recall (higher is better): Hamming {h.recall*100:.2f}% vs Euclidean {e.recall*100:.2f}%")

    print(f"\nFINAL CONCLUSION:")
    print(f"  {conclusion}")
    print("=" * 72)

    # 7. Persist derived thresholds to database if connected
    from iris_biometric.db import DatabaseStorage
    storage = DatabaseStorage()
    if storage.is_connected:
        storage.save_evaluation_thresholds(
            hamming_threshold=h.threshold,
            euclidean_threshold=e.threshold,
            data_source=result.data_source,
            n_comparisons=h.n_genuine + h.n_impostor,
        )
        print(f"\nDerived thresholds persisted to Neon Postgres:")
        print(f"  Hamming: {h.threshold:.4f}")
        print(f"  Euclidean: {e.threshold:.2f}")
        print(f"  Data source: {result.data_source}")
        print(f"  Comparisons: {h.n_genuine + h.n_impostor}")
    else:
        print(f"\nNote: Database not connected - thresholds not persisted.")
        print(f"Live identification will continue using provisional defaults from config.")

    # 7. Optionally save output to JSON
    if output_json:
        # Determine winner for JSON output
        if h.accuracy > e.accuracy:
            comparison_winner = "Hamming"
            accuracy_diff = (h.accuracy - e.accuracy) * 100
            comparison_conclusion = f"Hamming distance outperformed Euclidean distance on this evaluation set: {h.accuracy*100:.2f}% accuracy vs {e.accuracy*100:.2f}% accuracy (+{accuracy_diff:.2f} percentage points)"
        elif e.accuracy > h.accuracy:
            comparison_winner = "Euclidean"
            accuracy_diff = (e.accuracy - h.accuracy) * 100
            comparison_conclusion = f"Euclidean distance outperformed Hamming distance on this evaluation set: {e.accuracy*100:.2f}% accuracy vs {h.accuracy*100:.2f}% accuracy (+{accuracy_diff:.2f} percentage points)"
        else:
            comparison_winner = "Tie"
            comparison_conclusion = f"Hamming and Euclidean distances achieved identical accuracy on this evaluation set: {h.accuracy*100:.2f}%"

        report_data = {
            "data_source": result.data_source,
            "dataset_stats": stats,
            "gallery_enrolled_count": len(gallery.records),
            "skipped_gallery_count": len(skipped_gallery),
            "skipped_probe_count": len(result.skipped_probes),
            "comparison": {
                "winner": comparison_winner,
                "conclusion": comparison_conclusion,
                "accuracy_difference": abs(h.accuracy - e.accuracy) * 100,
            },
            "hamming": {
                "threshold": h.threshold,
                "eer": h.eer,
                "accuracy": h.accuracy,
                "precision": h.precision,
                "recall": h.recall,
                "f1": h.f1,
                "far": h.far,
                "frr": h.frr,
                "tp": h.tp,
                "tn": h.tn,
                "fp": h.fp,
                "fn": h.fn,
                "n_genuine": h.n_genuine,
                "n_impostor": h.n_impostor,
            },
            "euclidean": {
                "threshold": e.threshold,
                "eer": e.eer,
                "accuracy": e.accuracy,
                "precision": e.precision,
                "recall": e.recall,
                "f1": e.f1,
                "far": e.far,
                "frr": e.frr,
                "tp": e.tp,
                "tn": e.tn,
                "fp": e.fp,
                "fn": e.fn,
                "n_genuine": e.n_genuine,
                "n_impostor": e.n_impostor,
            },
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"\nSaved structured evaluation report to {output_json}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate iris biometric pipeline on MMU1 dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to MMU1 dataset directory (e.g. data/MMU-Iris-Database)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("mmu1_evaluation_report.json"),
        help="Path to save evaluation metrics JSON",
    )
    args = parser.parse_args()

    data_path = args.data_dir or find_default_mmu_dir()
    if not data_path:
        print("Error: Could not automatically locate MMU1 dataset folder.")
        print("Please specify the path using: python evaluate_mmu.py --data-dir <path/to/MMU-Iris-Database>")
        sys.exit(1)

    run_mmu_evaluation(data_path, output_json=args.output_json)
