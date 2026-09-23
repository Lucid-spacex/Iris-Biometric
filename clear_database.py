"""Clear all encrypted templates and evaluation data from Neon Postgres.

This script will:
1. Delete all encrypted iris templates
2. Delete the PCA projector
3. Delete evaluation thresholds
4. Reset the database to a clean state

WARNING: This action is irreversible and will delete all enrolled biometric data.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from iris_biometric.db import DatabaseStorage, Base, EncryptedTemplateModel, GalleryProjectorModel, EvaluationThresholdsModel


def clear_database():
    """Clear all biometric data from the database."""
    print("=" * 72)
    print("DATABASE CLEAR WARNING")
    print("=" * 72)
    print("This will DELETE all enrolled iris templates and evaluation data.")
    print("This action is IRREVERSIBLE.")
    print()
    
    confirm = input("Type 'YES' to confirm database deletion: ")
    if confirm.strip().upper() != "YES":
        print("Database clear cancelled.")
        return False
    
    storage = DatabaseStorage()
    
    if not storage.is_connected:
        print("ERROR: Database not connected. Cannot clear database.")
        return False
    
    try:
        with storage.Session() as session:
            # Delete all encrypted templates
            template_count = session.query(EncryptedTemplateModel).count()
            session.query(EncryptedTemplateModel).delete()
            print(f"Deleted {template_count} encrypted templates")
            
            # Delete PCA projector
            projector_count = session.query(GalleryProjectorModel).count()
            session.query(GalleryProjectorModel).delete()
            print(f"Deleted {projector_count} PCA projectors")
            
            # Delete evaluation thresholds
            threshold_count = session.query(EvaluationThresholdsModel).count()
            session.query(EvaluationThresholdsModel).delete()
            print(f"Deleted {threshold_count} evaluation threshold records")
            
            session.commit()
            
        print("=" * 72)
        print("DATABASE CLEARED SUCCESSFULLY")
        print("=" * 72)
        print("Your database is now clean and ready for fresh enrollment.")
        print("Please:")
        print("1. Set AES_MASTER_KEY_HEX in your .env file")
        print("2. Restart the application")
        print("3. Enroll new identities")
        print("=" * 72)
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to clear database: {e}")
        return False


if __name__ == "__main__":
    success = clear_database()
    sys.exit(0 if success else 1)