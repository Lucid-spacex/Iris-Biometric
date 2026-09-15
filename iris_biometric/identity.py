"""Stage 1 - Data input and identity handling.

Parses subject/eye/image identity from filenames following the CASIA
convention, and discovers image files under a directory. Kept separate
from any particular dataset so development fixtures and real CASIA data
share the same code path but are still distinguishable by type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .config import DEFAULT_CONFIG

# CASIA-style filename, e.g. "S1001L03.jpg" -> subject 1001, left eye, image 3.
_NAME_PATTERN = re.compile(r"S?(?P<subject>\d+)[_-]?(?P<eye>[LR])[_-]?(?P<index>\d+)", re.IGNORECASE)


class DataSource(Enum):
    """Distinguishes real evaluation data from development fixtures so the
    two can never be silently mixed in a report figure."""

    CASIA_STUDY_DATA = "casia_study_data"
    MMU1_STUDY_DATA = "mmu1_study_data"
    DEVELOPMENT_FIXTURE = "development_fixture"


@dataclass(frozen=True)
class IrisRecord:
    path: Path
    subject: str
    eye: str          # "L" or "R"
    index: str
    source: DataSource

    @property
    def identity_key(self) -> str:
        """Identity key of the form 'S<subject>-<eye>', per Section 4.3.1."""
        return f"S{self.subject}-{self.eye.upper()}"


class IdentityParseError(ValueError):
    """Raised when a filename does not match the CASIA or MMU1 naming convention."""


def parse_identity(path: Path, source: DataSource) -> IrisRecord:
    # 1. CASIA-style filename match (e.g. S1001L03.jpg)
    match = _NAME_PATTERN.search(path.stem)
    if match:
        return IrisRecord(
            path=path,
            subject=match.group("subject"),
            eye=match.group("eye").upper(),
            index=match.group("index"),
            source=source,
        )

    # 2. MMU1-style folder hierarchy match:
    # Expects .../<subject>/<left|right>/<filename>.bmp (or similar)
    parent_eye = path.parent.name.lower()
    if parent_eye in ("left", "right", "l", "r"):
        eye = "L" if parent_eye.startswith("l") else "R"
        subj_match = re.search(r"(\d+)", path.parent.parent.name)
        if subj_match:
            idx_match = re.search(r"(\d+)", path.stem)
            index = idx_match.group(1) if idx_match else path.stem
            return IrisRecord(
                path=path,
                subject=subj_match.group(1),
                eye=eye,
                index=index,
                source=source,
            )

    raise IdentityParseError(f"Cannot parse subject/eye/index from filename or path: {path}")


def discover_dataset(
    directory: Path,
    source: DataSource,
    extensions: tuple = DEFAULT_CONFIG.supported_extensions,
) -> list[IrisRecord]:
    """Walk `directory`, parse identity for every supported image file.

    Files that fail to parse are skipped rather than raising, so one badly
    named file cannot halt a whole dataset-wide run; callers that need to
    know about skipped files should call parse_identity() directly.
    """
    directory = Path(directory)
    records: list[IrisRecord] = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in extensions:
            continue
        try:
            records.append(parse_identity(path, source))
        except IdentityParseError:
            continue
    return records


def dataset_statistics(records: list[IrisRecord]) -> dict:
    """Descriptive statistics only - not a biometric result."""
    subjects = {r.subject for r in records}
    eyes = {r.identity_key for r in records}
    per_identity_counts = {}
    for r in records:
        per_identity_counts[r.identity_key] = per_identity_counts.get(r.identity_key, 0) + 1
    return {
        "n_images": len(records),
        "n_subjects": len(subjects),
        "n_identities": len(eyes),
        "min_images_per_identity": min(per_identity_counts.values()) if per_identity_counts else 0,
        "max_images_per_identity": max(per_identity_counts.values()) if per_identity_counts else 0,
    }
