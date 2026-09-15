"""End-to-end pipeline: enrolment -> encrypted gallery -> identification.

This is the glue module the demo app (and any future dataset-driven script)
calls into. It does not introduce any new biometric logic - it only
sequences the stage modules in the order Algorithm 1 / Algorithm 2 (Sec 3.4)
specify, and manages the AES-encrypted at-rest storage in between.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field

import numpy as np

from .config import DEFAULT_CONFIG
from .crypto import EncryptedTemplate, aes_decrypt, aes_encrypt, generate_key
from .db import DatabaseStorage
from .encoding import encode_iris
from .features import EuclideanProjector, euclidean_raw_vectors, select_hamming_feature
from .matching import IdentificationResult, identify as _identify
from .normalization import normalize_iris
from .segmentation import SegmentationResult, segment_iris
from .template import TemplateRecord, build_template


class PipelineStageFailure(ValueError):
    """Raised when an image fails a required stage (e.g. segmentation)."""


@dataclass
class ProcessedImage:
    identity_key: str
    hamming_feature: object          # features.HammingFeature
    raw_euclidean_vector: np.ndarray
    segmentation: SegmentationResult
    normalized_strip: np.ndarray     # for visualisation
    iris_code_bits: np.ndarray       # for visualisation


def process_image(image: np.ndarray, identity_key: str, config=DEFAULT_CONFIG) -> ProcessedImage:
    """Runs Stages 2-6 (segmentation through feature selection) on one image."""
    seg = segment_iris(image)
    if not seg.ok:
        raise PipelineStageFailure(f"Segmentation failed for {identity_key}: {seg.reason}")

    normalized = normalize_iris(image, seg.pupil, seg.iris, config=config)
    encoded = encode_iris(normalized, config=config)
    hamming_feature = select_hamming_feature(encoded)
    raw_vector = euclidean_raw_vectors([encoded], source=config.euclidean_feature_source)[0]

    return ProcessedImage(
        identity_key=identity_key,
        hamming_feature=hamming_feature,
        raw_euclidean_vector=raw_vector,
        segmentation=seg,
        normalized_strip=normalized.strip,
        iris_code_bits=encoded.iris_code,
    )


@dataclass
class EncryptedGallery:
    """An enrolled, AES-encrypted gallery plus the gallery-only-fitted
    Euclidean projector needed to score future probes consistently.

    Per Algorithm 1 ("store C with a non-reversible reference ID"),
    `records` is keyed by a random opaque token generated at enrolment -
    NOT by the identity_key. A random token has no computable relationship
    to the identity it points to, so even someone with raw access to the
    storage structure (no AES key) cannot tell whose record is whose from
    the key alone. The identity is recoverable only by decrypting the
    record's contents with the AES key.
    """

    aes_key: bytes
    projector: EuclideanProjector | None = None
    records: dict = field(default_factory=dict)  # reference_id -> EncryptedTemplate
    storage: DatabaseStorage | None = None

    def decrypt(self, reference_id: str) -> TemplateRecord:
        enc = self.records[reference_id]
        plaintext = aes_decrypt(enc, self.aes_key)
        return TemplateRecord.from_bytes(plaintext)

    def decrypt_all(self) -> list[TemplateRecord]:
        return [self.decrypt(k) for k in self.records]

    def identity_roster(self) -> list[str]:
        """Decrypts every record to list enrolled identities, in enrolment
        order. This is for display/bookkeeping only (e.g. a demo UI or an
        enrolment audit log with its own access controls) - the storage
        layer itself never indexes by identity."""
        return [t.identity_key for t in self.decrypt_all()]

    @classmethod
    def load_persisted(cls, aes_key: bytes, storage: DatabaseStorage | None = None) -> EncryptedGallery | None:
        """Attempts to reload enrolled templates and fitted PCA projector from Postgres."""
        store = storage or DatabaseStorage()
        if not store.is_connected:
            return None
        projector = store.load_projector()
        templates = store.load_all_templates()
        if templates:
            gallery = cls(aes_key=aes_key, projector=projector, records=templates, storage=store)
            try:
                # Test that the current AES key can decrypt the stored records
                first_ref = next(iter(gallery.records))
                gallery.decrypt(first_ref)
                return gallery
            except Exception:
                # Key mismatch (e.g. newly set AES_MASTER_KEY_HEX) - trigger fresh enrollment
                return None
        return None


def enroll_identity(
    identity_label: str,
    images: list[np.ndarray],
    gallery: EncryptedGallery,
    config=DEFAULT_CONFIG,
) -> tuple[list[str], ProcessedImage, list[str]]:
    """Enrolls one or more eye images for a given identity into the gallery.

    1. Runs full preprocessing pipeline (segmentation -> normalisation -> encoding -> feature selection).
    2. Initializes or updates the gallery's EuclideanProjector as needed.
    3. Builds AES-256 encrypted templates.
    4. Persists the new templates and updated projector into Postgres using
       non-reversible random reference IDs (zero plaintext identities).

    Returns:
        (enrolled_reference_ids, sample_processed_image, skipped_reasons)
    """
    if not images:
        raise ValueError("At least one image must be provided for enrollment.")

    processed: list[ProcessedImage] = []
    skipped: list[str] = []

    for idx, img in enumerate(images):
        try:
            proc = process_image(img, identity_key=identity_label, config=config)
            processed.append(proc)
        except PipelineStageFailure as exc:
            skipped.append(f"Image {idx+1}: {exc}")

    if not processed:
        raise PipelineStageFailure(
            f"Failed to segment the uploaded image(s): {'; '.join(skipped)}"
        )

    # Ensure gallery has a fitted Euclidean projector
    if gallery.projector is None:
        raw_matrix = np.stack([p.raw_euclidean_vector for p in processed])
        if len(raw_matrix) >= 2:
            gallery.projector = EuclideanProjector.fit(
                raw_matrix, n_components=min(config.pca_n_components, len(raw_matrix))
            )
        else:
            # Anchor single image with minimal perturbation to establish valid initial scale
            anchor = raw_matrix[0] + np.random.normal(0, 1e-5, size=raw_matrix[0].shape)
            gallery.projector = EuclideanProjector.fit(
                np.stack([raw_matrix[0], anchor]), n_components=2
            )

        if gallery.storage and gallery.storage.is_connected:
            gallery.storage.save_projector(gallery.projector)

    enrolled_ref_ids: list[str] = []
    for p in processed:
        record = build_template(
            identity_label, p.hamming_feature, gallery.projector, p.raw_euclidean_vector
        )
        encrypted = aes_encrypt(record.to_bytes(), gallery.aes_key)
        ref_id = secrets.token_hex(16)
        gallery.records[ref_id] = encrypted
        enrolled_ref_ids.append(ref_id)
        if gallery.storage and gallery.storage.is_connected:
            gallery.storage.save_template(ref_id, encrypted)

    return enrolled_ref_ids, processed[0], skipped


def enroll_gallery(
    images_with_identity: list[tuple[str, np.ndarray]],
    config=DEFAULT_CONFIG,
    aes_key_bits: int = DEFAULT_CONFIG.aes_key_bits,
    storage: DatabaseStorage | None = None,
) -> tuple[EncryptedGallery, list[str]]:
    """Stages 6 (gallery-only fit) + 9 (template build) + 10 (AES encrypt)
    for a batch of enrolment images.

    Returns the encrypted gallery plus a list of any identities skipped
    because segmentation failed (so a batch run never silently drops data).
    """
    processed: list[ProcessedImage] = []
    skipped: list[str] = []
    for identity_key, image in images_with_identity:
        try:
            processed.append(process_image(image, identity_key, config=config))
        except PipelineStageFailure:
            skipped.append(identity_key)

    if len(processed) < 2:
        raise PipelineStageFailure(
            "At least 2 successfully segmented enrolment images are required to fit the "
            "gallery-only Euclidean projector."
        )

    raw_matrix = np.stack([p.raw_euclidean_vector for p in processed])
    projector = EuclideanProjector.fit(raw_matrix, n_components=config.pca_n_components)

    aes_key = generate_key(aes_key_bits)
    db_store = storage if storage is not None else DatabaseStorage()
    gallery = EncryptedGallery(aes_key=aes_key, projector=projector, storage=db_store if db_store.is_connected else None)

    for p in processed:
        record = build_template(p.identity_key, p.hamming_feature, projector, p.raw_euclidean_vector)
        encrypted = aes_encrypt(record.to_bytes(), aes_key)
        reference_id = secrets.token_hex(16)  # non-reversible: random, not derived from identity
        gallery.records[reference_id] = encrypted

    # Persist templates and projector to database if connected
    if db_store.is_connected:
        db_store.save_templates_batch(gallery.records)
        db_store.save_projector(projector)

    return gallery, skipped


def identify_probe(
    image: np.ndarray,
    gallery: EncryptedGallery,
    config=DEFAULT_CONFIG,
    hamming_threshold: float | None = None,
    euclidean_threshold: float | None = None,
) -> tuple[IdentificationResult, ProcessedImage]:
    """Stages 6 (transform-only, no re-fit) + matching (11-12) for one probe.

    hamming_threshold / euclidean_threshold default to the provisional
    config placeholders, but can be overridden with thresholds derived by
    evaluation.find_eer() once an evaluation has been run.
    """
    if not gallery.records or gallery.projector is None:
        raise PipelineStageFailure(
            "The enrolled gallery is currently empty. Please enroll at least one person before identifying probes."
        )

    processed = process_image(image, identity_key="probe", config=config)

    probe_record = build_template(
        "probe",
        processed.hamming_feature,
        gallery.projector,
        processed.raw_euclidean_vector,
    )
    enrolled_records = gallery.decrypt_all()
    result = _identify(
        probe_record,
        enrolled_records,
        hamming_threshold=hamming_threshold if hamming_threshold is not None else config.provisional_hamming_threshold,
        euclidean_threshold=euclidean_threshold if euclidean_threshold is not None else config.provisional_euclidean_threshold,
    )
    return result, processed
