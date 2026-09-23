"""Flask web application for the iris biometric identification system.

Supports:
  - Real human enrollment (/enroll): extracts features, encrypts into AES-256
    templates, and stores in Neon Postgres under non-reversible reference IDs.
  - Probe identification (/identify or /): 1-to-many biometric matching using
    dual Hamming and Euclidean metrics.
  - Encrypted gallery audit (/gallery): inspects at-rest encrypted records and
    demonstrates that zero plaintext identities are exposed in database columns.

Synthetic demo routes (/demo/genuine, /demo/impostor, /demo/evaluate) have been
commented out as superseded by the real enrollment & identification pipeline.
"""

from __future__ import annotations

import io
import json
import os
import pathlib

import numpy as np
from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image

load_dotenv()

from iris_biometric.crypto import generate_key
from iris_biometric.db import DatabaseStorage
from iris_biometric.pipeline import (
    EncryptedGallery,
    PipelineStageFailure,
    enroll_identity,
    identify_probe,
)
from visualize import render_iris_code, render_segmentation_overlay, render_strip

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "iris-biometric-secure-key-2026")

_GALLERY: EncryptedGallery | None = None
_RUNTIME_AES_KEY: bytes | None = None


def get_master_key() -> bytes:
    """Returns the persistent 256-bit AES master key from .env or generates one."""
    global _RUNTIME_AES_KEY
    key_hex = os.environ.get("AES_MASTER_KEY_HEX", "").strip()
    if key_hex:
        try:
            return bytes.fromhex(key_hex)
        except ValueError:
            pass
    if _RUNTIME_AES_KEY is None:
        _RUNTIME_AES_KEY = generate_key(256)
    return _RUNTIME_AES_KEY


def get_gallery() -> EncryptedGallery:
    """Loads existing encrypted gallery from Neon Postgres, or initializes an empty gallery."""
    global _GALLERY
    if _GALLERY is None:
        storage = DatabaseStorage()
        master_key = get_master_key()

        # Try to reload existing gallery from Neon Postgres if available
        if storage.is_connected:
            persisted = EncryptedGallery.load_persisted(aes_key=master_key, storage=storage)
            if persisted:
                _GALLERY = persisted
                return _GALLERY

        # Initialize clean, empty gallery ready for real human enrollments
        _GALLERY = EncryptedGallery(
            aes_key=master_key,
            projector=None,
            records={},
            storage=storage if storage.is_connected else None,
        )
    return _GALLERY


def _ciphertext_for_identity(gallery: EncryptedGallery, identity_key: str) -> str | None:
    for ref_id, encrypted in gallery.records.items():
        try:
            if gallery.decrypt(ref_id).identity_key == identity_key:
                return encrypted.ciphertext[:48].hex()
        except Exception:
            continue
    return None


def _base_context(gallery: EncryptedGallery, active_page: str, **extra):
    is_postgres = gallery.storage is not None and gallery.storage.is_connected
    context = dict(
        active_page=active_page,
        gallery_count=len(gallery.records),
        storage_backend="Neon Postgres" if is_postgres else "In-Memory",
        pca_components=gallery.projector.n_components_effective if gallery.projector else 0,
        result=None,
    )
    context.update(extra)
    return context


# ==============================================================================
# ROUTE 1: IDENTIFY PROBE (SCANNER)
# ==============================================================================

@app.route("/", methods=["GET", "POST"])
@app.route("/identify", methods=["GET", "POST"])
def identify():
    gallery = get_gallery()
    if request.method == "GET":
        return render_template("identify.html", **_base_context(gallery, active_page="identify"))

    file = request.files.get("photo")
    if not file or file.filename == "":
        return render_template(
            "identify.html",
            **_base_context(gallery, active_page="identify", error="No image file selected."),
        )

    try:
        file_content = file.read()
        pil_img = Image.open(io.BytesIO(file_content)).convert("L")
        probe_img = np.array(pil_img)
        
        # Create base64 encoded version of original image for preview
        original_pil = Image.open(io.BytesIO(file_content))
        import base64
        from io import BytesIO
        buffered = BytesIO()
        original_pil.save(buffered, format="PNG")
        original_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        original_uri = f"data:image/png;base64,{original_base64}"
    except Exception as exc:
        return render_template(
            "identify.html",
            **_base_context(gallery, active_page="identify", error=f"Could not open image file: {exc}"),
        )

    label = f"Uploaded probe: {file.filename}"
    
    # Load thresholds before calling identify_probe
    from iris_biometric.config import DEFAULT_CONFIG
    
    # Try to load evaluation-derived thresholds from database
    h_thresh = DEFAULT_CONFIG.provisional_hamming_threshold
    e_thresh = DEFAULT_CONFIG.provisional_euclidean_threshold
    threshold_source = "provisional default, not yet evaluated"
    threshold_metadata = {}
    
    if gallery.storage and gallery.storage.is_connected:
        stored_thresholds = gallery.storage.load_evaluation_thresholds()
        if stored_thresholds:
            h_thresh = stored_thresholds["hamming_threshold"]
            e_thresh = stored_thresholds["euclidean_threshold"]
            threshold_source = f"derived from {stored_thresholds['data_source']}, N={stored_thresholds['n_comparisons']} comparisons"
            threshold_metadata = {
                "data_source": stored_thresholds["data_source"],
                "n_comparisons": stored_thresholds["n_comparisons"],
                "evaluation_date": stored_thresholds["evaluation_date"],
            }
    
    try:
        result, processed = identify_probe(
            probe_img, 
            gallery, 
            hamming_threshold=h_thresh,
            euclidean_threshold=e_thresh
        )
    except PipelineStageFailure as exc:
        return render_template(
            "identify.html",
            **_base_context(
                gallery,
                active_page="identify",
                error=f"Could not process image: {exc}",
                probe_label=label,
            ),
        )

    overlay_uri = render_segmentation_overlay(probe_img, processed.segmentation)
    strip_uri = render_strip(processed.normalized_strip)
    code_uri = render_iris_code(processed.iris_code_bits)

    winning_ciphertext_hex = None
    if result.best_hamming is not None:
        winning_ciphertext_hex = _ciphertext_for_identity(gallery, result.best_hamming.identity_key)

    candidates_sorted = sorted(result.candidates, key=lambda c: c.hamming_distance)

    return render_template(
        "result.html",
        **_base_context(
            gallery,
            active_page="identify",
            result=result,
            processed=processed,
            candidates=candidates_sorted,
            overlay_uri=overlay_uri,
            strip_uri=strip_uri,
            code_uri=code_uri,
            probe_label=label,
            winning_ciphertext_hex=winning_ciphertext_hex,
            hamming_threshold=h_thresh,
            euclidean_threshold=e_thresh,
            original_uri=original_uri,
            threshold_source=threshold_source,
            threshold_metadata=threshold_metadata,
        ),
    )


# ==============================================================================
# ROUTE 2: REAL BIOMETRIC ENROLLMENT
# ==============================================================================

@app.route("/enroll", methods=["GET", "POST"])
def enroll():
    gallery = get_gallery()
    if request.method == "GET":
        return render_template("enroll.html", **_base_context(gallery, active_page="enroll"))

    identity_name = request.form.get("identity_name", "").strip()
    if not identity_name:
        return render_template(
            "enroll.html",
            **_base_context(gallery, active_page="enroll", error="Please enter a subject name or identity label."),
        )

    uploaded_files = request.files.getlist("photos")
    if not uploaded_files or uploaded_files[0].filename == "":
        return render_template(
            "enroll.html",
            **_base_context(gallery, active_page="enroll", error="Please select at least one eye photograph."),
        )

    images: list[np.ndarray] = []
    for f in uploaded_files:
        try:
            pil_img = Image.open(io.BytesIO(f.read())).convert("L")
            images.append(np.array(pil_img))
        except Exception as exc:
            return render_template(
                "enroll.html",
                **_base_context(gallery, active_page="enroll", error=f"Invalid image '{f.filename}': {exc}"),
            )

    try:
        enrolled_ref_ids, sample_processed, skipped = enroll_identity(
            identity_label=identity_name,
            images=images,
            gallery=gallery,
        )
    except PipelineStageFailure as exc:
        return render_template(
            "enroll.html",
            **_base_context(
                gallery,
                active_page="enroll",
                error=str(exc),
                identity_name=identity_name,
            ),
        )

    # Render visualizations for the enrolled template
    first_img = images[0]
    overlay_uri = render_segmentation_overlay(first_img, sample_processed.segmentation)
    strip_uri = render_strip(sample_processed.normalized_strip)
    code_uri = render_iris_code(sample_processed.iris_code_bits)
    sample_ciphertext = gallery.records[enrolled_ref_ids[0]].ciphertext[:48].hex()

    return render_template(
        "enroll.html",
        **_base_context(
            gallery,
            active_page="enroll",
            enroll_success=True,
            identity_name=identity_name,
            enrolled_ref_ids=enrolled_ref_ids,
            sample_ciphertext=sample_ciphertext,
            overlay_uri=overlay_uri,
            strip_uri=strip_uri,
            code_uri=code_uri,
        ),
    )


# ==============================================================================
# ROUTE 3: ENCRYPTED GALLERY AUDIT
# ==============================================================================

@app.route("/gallery")
def gallery():
    gal = get_gallery()
    records_display = []
    for ref_id, enc in gal.records.items():
        try:
            decrypted = gal.decrypt(ref_id)
            identity_label = decrypted.identity_key
        except Exception:
            identity_label = "[Decryption Failed - Key Mismatch]"

        records_display.append(
            {
                "reference_id": ref_id,
                "ciphertext_hex": enc.ciphertext[:32].hex(),
                "identity_key": identity_label,
            }
        )

    return render_template(
        "gallery.html",
        **_base_context(gal, active_page="gallery", records=records_display),
    )


# ==============================================================================
# DEMO ROUTES (COMMENTED OUT AS SUPERSEDED BY REAL ENROLLMENT FLOW)
# ==============================================================================
# The three synthetic demo features below have been commented out because the
# system now uses real human enrollment (/enroll) and probe identification (/identify).
# They are preserved here in code comments so they can be restored or referenced later.
#
# @app.route("/demo/genuine", methods=["POST"])
# def demo_genuine():
#     # Previously simulated a re-scan of a synthetic identity
#     pass
#
# @app.route("/demo/impostor", methods=["POST"])
# def demo_impostor():
#     # Previously scanned an un-enrolled synthetic identity
#     pass
#
# @app.route("/demo/evaluate", methods=["POST"])
# def demo_evaluate():
#     # Previously generated synthetic EER/ROC evaluation numbers
#     pass


import traceback

@app.errorhandler(500)
def internal_error(error):
    """Show the real error message instead of a generic 500 page."""
    tb = traceback.format_exc()
    return f"""<!DOCTYPE html>
<html><head><title>500 Internal Server Error</title>
<style>body{{background:#1C1C21;color:#e2e8f0;font-family:monospace;padding:40px;}}
pre{{background:#25252C;padding:20px;border-radius:8px;overflow-x:auto;border-left:4px solid #ef4444;white-space:pre-wrap;}}
h1{{color:#ef4444;}}a{{color:#06b6d4;}}</style></head><body>
<h1>&#9888; Internal Server Error</h1>
<pre>{tb}</pre>
<p><a href="/">&#8592; Back to Home</a></p>
</body></html>""", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
