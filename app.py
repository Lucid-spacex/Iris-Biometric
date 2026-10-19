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
from iris_biometric.db import DatabaseStorage, EncryptedTemplateModel, GalleryProjectorModel
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
    """Returns the persistent 256-bit AES master key from .env.

    CRITICAL: If AES_MASTER_KEY_HEX is not set in .env, the application
    will exit with an error. This prevents the recurring AES key mismatch
    problem where templates become undecryptable on server restart.
    """
    global _RUNTIME_AES_KEY
    key_hex = os.environ.get("AES_MASTER_KEY_HEX", "").strip()
    if key_hex:
        try:
            return bytes.fromhex(key_hex)
        except ValueError:
            print("=" * 80)
            print("FATAL ERROR: AES_MASTER_KEY_HEX is set but invalid.")
            print(f"Value: {key_hex[:20]}...")
            print("=" * 80)
            raise SystemExit(1)

    # No valid key in environment - FATAL ERROR
    print("=" * 80)
    print("FATAL ERROR: AES_MASTER_KEY_HEX not set in .env file or environment.")
    print("=" * 80)
    print("The application cannot start without a persistent AES key.")
    print("To fix this:")
    print("  1. Copy .env.example to .env in the deploy folder")
    print("  2. Generate a key: python -c \"import secrets; print(secrets.token_hex(32))\"")
    print("  3. Set AES_MASTER_KEY_HEX=your_key_hex_value in .env")
    print("=" * 80)
    raise SystemExit(1)


def get_gallery(force_reload: bool = False) -> EncryptedGallery:
    """Loads existing encrypted gallery from Neon Postgres, or initializes an empty gallery.

    If the database is connected and has a different record count than the in-memory
    cache (e.g. after ingest_mmu.py runs externally), it automatically re-syncs with Postgres.
    """
    global _GALLERY
    storage = DatabaseStorage()
    master_key = get_master_key()

    if storage.is_connected:
        try:
            with storage.Session() as session:
                db_count = session.query(EncryptedTemplateModel).count()
            if _GALLERY is None or force_reload or len(_GALLERY.records) != db_count:
                persisted = EncryptedGallery.load_persisted(aes_key=master_key, storage=storage)
                if persisted is not None:
                    _GALLERY = persisted
                    return _GALLERY
                else:
                    print(f"[WARNING] Database has {db_count} templates but failed to load (likely AES key mismatch or decryption error)")
        except Exception as e:
            print(f"[WARNING] Error loading gallery from database: {e}")
            import traceback
            traceback.print_exc()
            # Fall through to in-memory gallery below

    if _GALLERY is None or force_reload:
        if not storage.is_connected:
            print("[WARNING] Database not connected - using IN-MEMORY storage. Data will be lost on restart.")
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
    except ValueError as exc:
        if "AES master key" in str(exc) or "Invalid padding" in str(exc):
            return render_template(
                "identify.html",
                **_base_context(
                    gallery,
                    active_page="identify",
                    error=f"Decryption error: {exc}. Please check your AES_MASTER_KEY_HEX environment variable matches the key used during enrollment.",
                    probe_label=label,
                ),
            )
        raise

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
    except ValueError as exc:
        if "AES master key" in str(exc) or "Invalid padding" in str(exc):
            return render_template(
                "enroll.html",
                **_base_context(
                    gallery,
                    active_page="enroll",
                    error=f"Decryption error: {exc}. Please check your AES_MASTER_KEY_HEX environment variable matches the key used during enrollment.",
                    identity_name=identity_name,
                ),
            )
        raise

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

@app.route("/evaluate", methods=["GET", "POST"])
def evaluate():
    """Run evaluation against the MMU1 dataset using the standard evaluation pipeline."""
    gallery = get_gallery()
    
    if request.method == "GET":
        return render_template(
            "evaluate.html",
            **_base_context(gallery, active_page="evaluate"),
        )
    
    # POST request - run the evaluation
    try:
        from iris_biometric.evaluation import evaluate as run_evaluation
        from iris_biometric.identity import discover_dataset, DataSource, partition_dataset
        from iris_biometric.config import DEFAULT_CONFIG
        from iris_biometric.pipeline import enroll_gallery
        from pathlib import Path
        import numpy as np
        from PIL import Image
        import time
        
        print("Starting evaluation request...")
        start_time = time.time()
        
        # Get MMU1 dataset path - check common locations
        dataset_path = None
        possible_paths = [
            Path(r"C:\Users\Samuel\.cache\kagglehub\datasets\naureenmohammad\mmu-iris-dataset\versions\1\MMU-Iris-Database"),
            Path("data/MMU-Iris-Database"),
            Path("MMU-Iris-Database"),
            Path.home() / "Downloads" / "MMU-Iris-Database",
        ]
        
        for path in possible_paths:
            if path.exists():
                dataset_path = path
                break
        
        if not dataset_path:
            return render_template(
                "evaluate.html",
                **_base_context(gallery, active_page="evaluate", error="MMU1 dataset not found. Please download the MMU1 Iris Database and place it in data/MMU-Iris-Database or the expected cache location."),
            )
        
        # Discover records using the existing working function
        records = discover_dataset(dataset_path, source=DataSource.MMU1_STUDY_DATA)
        
        if not records:
            return render_template(
                "evaluate.html",
                **_base_context(gallery, active_page="evaluate", error=f"No valid iris images found under {dataset_path}"),
            )
        
        # Partition using the function we just added to identity.py
        gallery_recs, probe_recs = partition_dataset(records)
        
        # Load images for gallery enrollment
        gallery_images = []
        for rec in gallery_recs:
            try:
                img = Image.open(rec.path).convert("L")
                img_array = np.array(img)
                gallery_images.append((rec.identity_key, img_array))
            except Exception as e:
                print(f"Warning: could not read {rec.path}: {e}")
        
        if not gallery_images:
            return render_template(
                "evaluate.html",
                **_base_context(gallery, active_page="evaluate", error="Insufficient gallery images for evaluation."),
            )
        
        print(f"Loaded {len(gallery_images)} gallery images for evaluation")
        
        # Enroll gallery (this creates a new ISOLATED gallery for evaluation only).
        # storage=None is mandatory here: we must NOT write MMU1 evaluation
        # templates into the live encrypted_templates table, which would
        # overwrite enrolled human subjects with templates encrypted under a
        # throwaway AES key, breaking the production gallery.
        eval_gallery, skipped_gallery = enroll_gallery(gallery_images, storage=None)
        print(f"Enrolled {len(eval_gallery.records)} templates (skipped {len(skipped_gallery)})")
        
        # Load images for probes
        probe_images = []
        for rec in probe_recs:
            try:
                img = Image.open(rec.path).convert("L")
                img_array = np.array(img)
                probe_images.append((rec.identity_key, img_array))
            except Exception as e:
                print(f"Warning: could not read {rec.path}: {e}")
        
        if not probe_images:
            return render_template(
                "evaluate.html",
                **_base_context(gallery, active_page="evaluate", error="Insufficient probe images for evaluation. Each identity needs at least 2 images."),
            )
        
        print(f"Loaded {len(probe_images)} probe images for evaluation")
        
        # Run evaluation using the existing working function
        result = run_evaluation(
            gallery=eval_gallery,
            probes=probe_images,
            data_source="MMU1 Iris Database",
            config=DEFAULT_CONFIG,
        )
        
        # Display results
        h = result.hamming
        e = result.euclidean
        
        results_html = f"""
        <div class="panel">
            <h2>Evaluation Results - MMU1 Iris Database</h2>
            <p class="sub">Biometric evaluation using standard MMU1 dataset partitioning</p>
            
            <div class="stat-row">
                <div class="stat">
                    <div class="num">{h.accuracy*100:.2f}%</div>
                    <div class="label">Hamming Accuracy</div>
                </div>
                <div class="stat">
                    <div class="num">{e.accuracy*100:.2f}%</div>
                    <div class="label">Euclidean Accuracy</div>
                </div>
                <div class="stat">
                    <div class="num">{h.eer*100:.2f}%</div>
                    <div class="label">Hamming EER</div>
                </div>
                <div class="stat">
                    <div class="num">{e.eer*100:.2f}%</div>
                    <div class="label">Euclidean EER</div>
                </div>
            </div>
            
            <h3>Detailed Metrics</h3>
            <table class="scores">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Hamming</th>
                        <th>Euclidean</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Threshold</td>
                        <td>{h.threshold:.4f}</td>
                        <td>{e.threshold:.2f}</td>
                    </tr>
                    <tr>
                        <td>Accuracy</td>
                        <td>{h.accuracy*100:.2f}%</td>
                        <td>{e.accuracy*100:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Precision</td>
                        <td>{h.precision*100:.2f}%</td>
                        <td>{e.precision*100:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Recall</td>
                        <td>{h.recall*100:.2f}%</td>
                        <td>{e.recall*100:.2f}%</td>
                    </tr>
                    <tr>
                        <td>F1 Score</td>
                        <td>{h.f1:.4f}</td>
                        <td>{e.f1:.4f}</td>
                    </tr>
                    <tr>
                        <td>FAR</td>
                        <td>{h.far*100:.2f}%</td>
                        <td>{e.far*100:.2f}%</td>
                    </tr>
                    <tr>
                        <td>FRR</td>
                        <td>{h.frr*100:.2f}%</td>
                        <td>{e.frr*100:.2f}%</td>
                    </tr>
                </tbody>
            </table>
            
            <h3>Comparison Counts</h3>
            <p>Hamming: TP={h.tp}, TN={h.tn}, FP={h.fp}, FN={h.fn} (Genuine={h.n_genuine}, Impostor={h.n_impostor})</p>
            <p>Euclidean: TP={e.tp}, TN={e.tn}, FP={e.fp}, FN={e.fn} (Genuine={e.n_genuine}, Impostor={e.n_impostor})</p>
            
            <h3>Performance Comparison</h3>
        """
        
        # Add comparison conclusion
        if h.accuracy > e.accuracy:
            winner = "Hamming"
            accuracy_diff = (h.accuracy - e.accuracy) * 100
            conclusion = f"Hamming distance outperformed Euclidean distance on this MMU1 evaluation: {h.accuracy*100:.2f}% accuracy vs {e.accuracy*100:.2f}% accuracy (+{accuracy_diff:.2f} percentage points)"
        elif e.accuracy > h.accuracy:
            winner = "Euclidean"
            accuracy_diff = (e.accuracy - h.accuracy) * 100
            conclusion = f"Euclidean distance outperformed Hamming distance on this MMU1 evaluation: {e.accuracy*100:.2f}% accuracy vs {h.accuracy*100:.2f}% accuracy (+{accuracy_diff:.2f} percentage points)"
        else:
            winner = "Tie"
            conclusion = f"Hamming and Euclidean distances achieved identical accuracy on this MMU1 evaluation: {h.accuracy*100:.2f}%"
        
        results_html += f"""
            <p><strong>Winner:</strong> {winner}</p>
            <p><strong>Conclusion:</strong> {conclusion}</p>
            
            <div class="form-actions">
                <a href="{{ url_for('identify') }}" class="btn btn-secondary">Back to Identification</a>
            </div>
        </div>
        """
        
        # Persist derived thresholds to database if connected
        if gallery.storage and gallery.storage.is_connected:
            gallery.storage.save_evaluation_thresholds(
                hamming_threshold=h.threshold,
                euclidean_threshold=e.threshold,
                data_source="MMU1 Iris Database",
                n_comparisons=h.n_genuine + h.n_impostor,
            )
        
        elapsed = time.time() - start_time
        print(f"Evaluation completed in {elapsed:.2f} seconds")
        
        return render_template(
            "evaluate.html",
            **_base_context(gallery, active_page="evaluate"),
            custom_content=results_html
        )
        
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        return render_template(
            "evaluate.html",
            **_base_context(gallery, active_page="evaluate", error=f"Evaluation failed: {exc}\n\n{tb}"),
        )


@app.route("/gallery")
def gallery():
    gal = get_gallery(force_reload=True)
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


# Startup verification: check database connectivity and AES key before serving
print("=" * 80)
print("STARTUP VERIFICATION")
print("=" * 80)
try:
    storage = DatabaseStorage()
    if storage.is_connected:
        with storage.Session() as session:
            db_count = session.query(EncryptedTemplateModel).count()
            projector_count = session.query(GalleryProjectorModel).count()
        print(f"[OK] Database connected: Neon Postgres")
        print(f"[OK] Database contains: {db_count} encrypted templates, {projector_count} projector(s)")
        master_key = get_master_key()
        print(f"[OK] AES_MASTER_KEY_HEX is set (length: {len(master_key)} bytes)")
    else:
        print("[WARNING] Database NOT connected - running in IN-MEMORY mode")
        print("  Data will be lost on server restart. Check DATABASE_URL environment variable.")
except Exception as e:
    print(f"[ERROR] during startup verification: {e}")
    import traceback
    traceback.print_exc()
    print("=" * 80)
    raise SystemExit(1)
print("=" * 80)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
