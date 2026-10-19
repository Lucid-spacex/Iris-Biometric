"""Postgres database persistence layer for encrypted templates and PCA projector.

Implements SQLAlchemy ORM models and storage operations with privacy-preserving
design:
  - Enrolled templates are stored ONLY by a random, non-reversible reference ID.
  - Identity strings exist strictly inside the AES-encrypted ciphertext payload.
  - The gallery-fitted PCA projector (EuclideanProjector) is serialized and
    stored in Postgres so Render web server restarts can reload the exact
    projection matrix without losing alignment with enrolled templates.
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    LargeBinary,
    String,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from .crypto import EncryptedTemplate
from .features import EuclideanProjector

Base = declarative_base()


class EncryptedTemplateModel(Base):
    """Encrypted template table.

    Security principle: Contains ZERO plaintext identity information.
    The primary key is a randomly generated 128-bit hex token (reference_id).
    The person's identity is recoverable ONLY by decrypting the ciphertext
    with the AES-256 secret key.
    """

    __tablename__ = "encrypted_templates"

    reference_id = Column(String(64), primary_key=True)
    ciphertext = Column(LargeBinary, nullable=False)
    iv = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class GalleryProjectorModel(Base):
    """Persisted fitted PCA projector for consistent probe transforms across restarts."""

    __tablename__ = "gallery_projector"

    id = Column(Integer, primary_key=True, default=1)
    projector_bytes = Column(LargeBinary, nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class EvaluationThresholdsModel(Base):
    """Persisted evaluation-derived thresholds for real biometric matching."""

    __tablename__ = "evaluation_thresholds"

    id = Column(Integer, primary_key=True, default=1)
    hamming_threshold = Column(Integer, nullable=False)  # Stored as integer to avoid floating precision issues
    euclidean_threshold = Column(Integer, nullable=False)  # Stored as integer to avoid floating precision issues
    data_source = Column(String(255), nullable=False)
    n_comparisons = Column(Integer, nullable=False)
    evaluation_date = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


def normalize_db_url(url: str) -> str:
    """Normalize connection strings (e.g. postgres:// -> postgresql://)."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql://", 1)
    return url


def get_db_engine(database_url: Optional[str] = None):
    url = database_url or os.environ.get("DATABASE_URL")
    if not url:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            url = os.environ.get("DATABASE_URL")
        except Exception:
            pass
    if not url:
        return None
    url = normalize_db_url(url)
    # pool_pre_ping ensures connections disconnected by Neon when idle are reconnected cleanly
    return create_engine(url, pool_pre_ping=True, pool_recycle=300)


def init_db(engine) -> None:
    """Create tables if they do not exist."""
    if engine is not None:
        Base.metadata.create_all(engine)


class DatabaseStorage:
    """Manages database read/write operations for EncryptedGallery."""

    def __init__(self, database_url: Optional[str] = None):
        self.engine = get_db_engine(database_url)
        if self.engine is not None:
            try:
                init_db(self.engine)
                self.Session = sessionmaker(bind=self.engine)
            except Exception as e:
                # Connection failed - this can happen due to network issues
                # Set Session to None so is_connected returns False
                print(f"[WARNING] Database connection failed during initialization: {e}")
                self.Session = None
        else:
            self.Session = None

    @property
    def is_connected(self) -> bool:
        return self.Session is not None

    def save_template(self, reference_id: str, encrypted: EncryptedTemplate) -> None:
        if not self.is_connected:
            return
        with self.Session() as session:
            model = session.query(EncryptedTemplateModel).filter_by(reference_id=reference_id).first()
            if not model:
                model = EncryptedTemplateModel(
                    reference_id=reference_id,
                    ciphertext=encrypted.ciphertext,
                    iv=encrypted.iv,
                )
                session.add(model)
            else:
                model.ciphertext = encrypted.ciphertext
                model.iv = encrypted.iv
            session.commit()

    def save_templates_batch(self, records: dict[str, EncryptedTemplate]) -> None:
        if not self.is_connected or not records:
            return
        with self.Session() as session:
            for ref_id, enc in records.items():
                existing = session.query(EncryptedTemplateModel).filter_by(reference_id=ref_id).first()
                if existing:
                    existing.ciphertext = enc.ciphertext
                    existing.iv = enc.iv
                else:
                    session.add(
                        EncryptedTemplateModel(
                            reference_id=ref_id,
                            ciphertext=enc.ciphertext,
                            iv=enc.iv,
                        )
                    )
            session.commit()

    def load_all_templates(self) -> dict[str, EncryptedTemplate]:
        if not self.is_connected:
            return {}
        with self.Session() as session:
            rows = session.query(EncryptedTemplateModel).all()
            return {
                row.reference_id: EncryptedTemplate(ciphertext=row.ciphertext, iv=row.iv)
                for row in rows
            }

    def save_projector(self, projector: EuclideanProjector) -> None:
        if not self.is_connected:
            return
        data_bytes = pickle.dumps(projector)
        with self.Session() as session:
            row = session.query(GalleryProjectorModel).filter_by(id=1).first()
            if not row:
                row = GalleryProjectorModel(id=1, projector_bytes=data_bytes)
                session.add(row)
            else:
                row.projector_bytes = data_bytes
            session.commit()

    def load_projector(self) -> Optional[EuclideanProjector]:
        if not self.is_connected:
            return None
        with self.Session() as session:
            row = session.query(GalleryProjectorModel).filter_by(id=1).first()
            if row and row.projector_bytes:
                try:
                    return pickle.loads(row.projector_bytes)
                except Exception as e:
                    print(f"Warning: Failed to deserialize projector: {e}")
                    return None
        return None

    def save_evaluation_thresholds(
        self,
        hamming_threshold: float,
        euclidean_threshold: float,
        data_source: str,
        n_comparisons: int,
    ) -> None:
        """Save evaluation-derived thresholds to database."""
        if not self.is_connected:
            return
        # Store as integers to avoid floating point precision issues
        # Hamming: multiply by 10000 for 4 decimal places
        # Euclidean: multiply by 10 for 1 decimal place
        with self.Session() as session:
            row = session.query(EvaluationThresholdsModel).filter_by(id=1).first()
            if not row:
                row = EvaluationThresholdsModel(
                    id=1,
                    hamming_threshold=int(hamming_threshold * 10000),
                    euclidean_threshold=int(euclidean_threshold * 10),
                    data_source=data_source,
                    n_comparisons=n_comparisons,
                )
                session.add(row)
            else:
                row.hamming_threshold = int(hamming_threshold * 10000)
                row.euclidean_threshold = int(euclidean_threshold * 10)
                row.data_source = data_source
                row.n_comparisons = n_comparisons
                row.evaluation_date = func.now()
            session.commit()

    def load_evaluation_thresholds(self) -> Optional[dict]:
        """Load evaluation-derived thresholds from database."""
        if not self.is_connected:
            return None
        with self.Session() as session:
            row = session.query(EvaluationThresholdsModel).filter_by(id=1).first()
            if row:
                # Convert back from integer storage
                return {
                    "hamming_threshold": row.hamming_threshold / 10000.0,
                    "euclidean_threshold": row.euclidean_threshold / 10.0,
                    "data_source": row.data_source,
                    "n_comparisons": row.n_comparisons,
                    "evaluation_date": row.evaluation_date,
                }
        return None
