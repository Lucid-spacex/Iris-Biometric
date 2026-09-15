"""Helper script to obtain and extract the MMU1 Iris Dataset.

Supports:
1. Automatic detection of downloaded zip files in ~/Downloads or the project folder.
2. Direct Kaggle API download if Kaggle credentials are configured.
3. Interactive guidance for downloading from Kaggle.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import zipfile

DATA_DIR = pathlib.Path("data")
MMU_DIR = DATA_DIR / "MMU-Iris-Database"


def extract_zip(zip_path: pathlib.Path, target_dir: pathlib.Path) -> bool:
    print(f"Extracting {zip_path.resolve()} to {target_dir.resolve()}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)
    print("Extraction complete!")
    return True


def find_zip_in_downloads() -> pathlib.Path | None:
    downloads = pathlib.Path.home() / "Downloads"
    current = pathlib.Path(".")
    patterns = ["*mmu*.zip", "*iris*.zip"]
    candidates = []
    for d in [downloads, current, DATA_DIR]:
        if d.exists():
            for p in patterns:
                candidates.extend(d.glob(p))
    # Exclude zip of the source code itself
    filtered = [c for c in candidates if "demo" not in c.name.lower()]
    return filtered[0] if filtered else None


def download_via_kaggle(target_dir: pathlib.Path) -> bool:
    print("Attempting download via Kaggle API...")
    try:
        cmd = [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            "naureenmohammad/mmu-iris-dataset",
            "-p",
            str(target_dir),
            "--unzip",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("Successfully downloaded and unzipped MMU1 dataset via Kaggle CLI!")
            return True
        else:
            print(f"Kaggle CLI returned non-zero status:\n{result.stderr or result.stdout}")
            return False
    except Exception as e:
        print(f"Error invoking Kaggle CLI: {e}")
        return False


def main():
    print("=" * 70)
    print("MMU1 IRIS DATASET DOWNLOAD & SETUP HELPER")
    print("=" * 70)

    # Check if dataset is already extracted and present
    if MMU_DIR.exists() and any(MMU_DIR.rglob("*.bmp")):
        bmp_count = len(list(MMU_DIR.rglob("*.bmp")))
        print(f"MMU1 dataset is already extracted in: {MMU_DIR.resolve()}")
        print(f"Found {bmp_count} .bmp images.")
        print("\nYou can run evaluation now using:")
        print(f"  python evaluate_mmu.py --data-dir \"{MMU_DIR}\"")
        return

    # Check if a downloaded zip file is in Downloads or current folder
    found_zip = find_zip_in_downloads()
    if found_zip:
        print(f"Found downloaded dataset archive: {found_zip.resolve()}")
        extract_zip(found_zip, MMU_DIR)
        bmp_count = len(list(MMU_DIR.rglob("*.bmp")))
        print(f"Found {bmp_count} .bmp images.")
        print("\nReady! You can run evaluation now using:")
        print(f"  python evaluate_mmu.py --data-dir \"{MMU_DIR}\"")
        return

    # Check if Kaggle credentials are set up
    kaggle_json = pathlib.Path.home() / ".kaggle" / "kaggle.json"
    kaggle_token = pathlib.Path.home() / ".kaggle" / "access_token"
    has_kaggle_auth = (
        kaggle_json.exists()
        or kaggle_token.exists()
        or ("KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ)
        or "KAGGLE_API_TOKEN" in os.environ
    )

    if has_kaggle_auth:
        print("Detected Kaggle credentials. Downloading automatically...")
        if download_via_kaggle(MMU_DIR):
            print("\nReady! You can run evaluation now using:")
            print(f"  python evaluate_mmu.py --data-dir \"{MMU_DIR}\"")
            return

    # If neither, guide the user
    print("\nThe MMU1 dataset (~14 MB) is not yet downloaded.")
    print("\nPlease choose one of the two easy options below:")
    print("\n--- OPTION 1: Direct Browser Download (Easiest) ---")
    print("1. Go to: https://www.kaggle.com/datasets/naureenmohammad/mmu-iris-dataset")
    print("2. Click the 'Download' button in Kaggle.")
    print("3. Keep the downloaded zip in your Downloads folder, and re-run this script:")
    print("   python download_mmu.py")
    print("   (It will automatically detect and extract it for you!)")
    print("\n--- OPTION 2: Kaggle API Token ---")
    print("1. Go to https://www.kaggle.com/settings/api")
    print("2. Click 'Generate New Token' to download 'kaggle.json'.")
    print(f"3. Place 'kaggle.json' in: {pathlib.Path.home() / '.kaggle'}")
    print("4. Re-run: python download_mmu.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
