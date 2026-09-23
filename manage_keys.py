"""Utility for managing AES master keys for the iris biometric system.

This script helps with:
1. Generating a new AES-256 master key
2. Displaying the current key from environment
3. Checking database key compatibility
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from iris_biometric.crypto import generate_key


def generate_new_key():
    """Generate a new AES-256 master key and display it."""
    key = generate_key(256)
    key_hex = key.hex()
    print("=" * 72)
    print("NEW AES-256 MASTER KEY GENERATED")
    print("=" * 72)
    print(f"Key (hex): {key_hex}")
    print(f"Key length: {len(key)} bytes ({len(key) * 8} bits)")
    print()
    print("IMPORTANT:")
    print("1. Save this key securely - you'll need it to decrypt existing templates")
    print("2. Set this as AES_MASTER_KEY_HEX in your .env file or Render environment")
    print("3. If you change this key, existing encrypted templates will become unreadable")
    print("4. To use a new key with existing data, you must re-enroll all templates")
    print()
    print("For Render deployment, add this environment variable:")
    print(f"  AES_MASTER_KEY_HEX={key_hex}")
    print("=" * 72)
    return key_hex


def check_current_key():
    """Check and display the current AES master key from environment."""
    key_hex = os.environ.get("AES_MASTER_KEY_HEX", "").strip()
    print("=" * 72)
    print("CURRENT AES MASTER KEY STATUS")
    print("=" * 72)
    
    if key_hex:
        try:
            key = bytes.fromhex(key_hex)
            print(f"Key is set: {key_hex[:16]}...{key_hex[-16:]}")  # Show first/last 16 chars
            print(f"Key length: {len(key)} bytes ({len(key) * 8} bits)")
            
            if len(key) == 32:
                print("Status: Valid AES-256 key")
            elif len(key) == 24:
                print("Status: Valid AES-192 key")
            elif len(key) == 16:
                print("Status: Valid AES-128 key")
            else:
                print(f"Status: Invalid key length (expected 16, 24, or 32 bytes)")
        except ValueError:
            print("Status: Invalid hex format")
    else:
        print("Status: No AES_MASTER_KEY_HEX environment variable set")
        print("The system will generate a random key on each startup")
        print("WARNING: This means templates will not be decryptable across restarts")
    
    print("=" * 72)


def clear_database_warning():
    """Print warning about clearing database."""
    print("=" * 72)
    print("DATABASE CLEAR WARNING")
    print("=" * 72)
    print("If you're experiencing 'Invalid padding' errors, you have two options:")
    print()
    print("OPTION 1: Use the original key")
    print("  - Find the AES_MASTER_KEY_HEX used during original enrollment")
    print("  - Set it in your .env file or Render environment variables")
    print("  - Restart the application")
    print()
    print("OPTION 2: Clear database and start fresh")
    print("  - This will DELETE all enrolled templates")
    print("  - You'll need to re-enroll all identities")
    print("  - Generate a new key using: python manage_keys.py --generate")
    print("  - Clear the database tables")
    print("  - Re-enroll your data")
    print()
    print("To clear Neon Postgres database, you can:")
    print("  1. Use the Neon console to drop/recreate tables")
    print("  2. Or use a SQL client to TRUNCATE the encrypted_templates table")
    print("=" * 72)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage AES master keys for iris biometric system")
    parser.add_argument("--generate", action="store_true", help="Generate a new AES-256 master key")
    parser.add_argument("--check", action="store_true", help="Check current key status")
    parser.add_argument("--clear-warning", action="store_true", help="Show database clear warning")
    
    args = parser.parse_args()
    
    if args.generate:
        generate_new_key()
    elif args.check:
        check_current_key()
    elif args.clear_warning:
        clear_database_warning()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()