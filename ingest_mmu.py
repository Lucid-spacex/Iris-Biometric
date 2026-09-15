"""Script to ingest the MMU1 dataset into the Neon Postgres database."""

import glob
import os
import numpy as np
from PIL import Image

from iris_biometric.pipeline import enroll_gallery
from iris_biometric.db import DatabaseStorage, EncryptedTemplateModel, GalleryProjectorModel
from app import get_master_key

def main():
    dataset_path = r"C:\Users\Samuel\.cache\kagglehub\datasets\naureenmohammad\mmu-iris-dataset\versions\1\MMU-Iris-Database"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    images_with_identity = []
    subjects = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    print(f"Scanning {len(subjects)} subjects in MMU1 dataset...")
    for subject_id in sorted(subjects):
        for eye in ['left', 'right']:
            eye_path = os.path.join(dataset_path, subject_id, eye)
            if not os.path.exists(eye_path):
                continue
                
            bmp_files = sorted(glob.glob(os.path.join(eye_path, "*.bmp")))
            enrollment_files = bmp_files[:3] # Use first 3 for enrollment
            
            eye_code = 'L' if eye == 'left' else 'R'
            identity_key = f"MMU1_{int(subject_id):02d}_{eye_code}"
            
            for file_path in enrollment_files:
                try:
                    pil_img = Image.open(file_path).convert("L")
                    img_array = np.array(pil_img)
                    images_with_identity.append((identity_key, img_array))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    print(f"Loaded {len(images_with_identity)} images for {len(subjects) * 2} identities.")
    
    storage = DatabaseStorage()
    if not storage.is_connected:
        print("Error: Could not connect to Neon Postgres.")
        return
        
    # Purge existing data to initialize a clean global PCA model
    print("Purging existing database records...")
    with storage.Session() as session:
        session.query(EncryptedTemplateModel).delete()
        session.query(GalleryProjectorModel).delete()
        session.commit()

    print("Running batch enrollment pipeline (this may take a minute due to segmentation & PCA)...")
    aes_key = get_master_key()
    
    gallery, skipped = enroll_gallery(
        images_with_identity=images_with_identity,
        aes_key=aes_key,
        storage=storage
    )
    
    print(f"\nEnrollment complete.")
    print(f"Successfully enrolled: {len(gallery.records)} templates.")
    if skipped:
        print(f"Skipped {len(skipped)} images due to segmentation failures.")

if __name__ == "__main__":
    main()
