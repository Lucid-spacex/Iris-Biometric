"""Stage 10 - AES encryption/decryption of templates (Algorithm 3, Sec 3.4).

Implements FIPS 197 AES in CBC mode (NIST SP 800-38A), matching the design
in the report exactly:
  1. Select AES key length (128/192/256 bits).
  2. Generate a fresh random IV per template; pad to the 128-bit block size.
  3. Encrypt: C = AES-CBC(T, K, IV); store IV alongside C.
  4. Decrypt: T = AES-CBC^-1(C, K, IV); remove padding to recover T.

Uses the `cryptography` library rather than PyCryptodome (named in the
report), because this build environment has no network access to install
new packages and PyCryptodome is not preinstalled. Both implement the same
NIST-standardised primitive (AES-CBC per FIPS 197 / SP 800-38A) - this is a
library substitution, not an algorithm change, and is worth noting
explicitly if this code is folded into the report.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

_BLOCK_BITS = 128
_KEY_BYTES_BY_BITS = {128: 16, 192: 24, 256: 32}


class UnsupportedKeyLength(ValueError):
    pass


def generate_key(key_bits: int = 256) -> bytes:
    if key_bits not in _KEY_BYTES_BY_BITS:
        raise UnsupportedKeyLength(f"AES key length must be 128, 192, or 256 bits, got {key_bits}")
    return os.urandom(_KEY_BYTES_BY_BITS[key_bits])


@dataclass(frozen=True)
class EncryptedTemplate:
    ciphertext: bytes
    iv: bytes


def aes_encrypt(plaintext: bytes, key: bytes) -> EncryptedTemplate:
    if len(key) not in _KEY_BYTES_BY_BITS.values():
        raise UnsupportedKeyLength(f"Key must be 16, 24, or 32 bytes, got {len(key)}")

    iv = os.urandom(16)  # 128-bit IV, fresh per template
    padder = padding.PKCS7(_BLOCK_BITS).padder()
    padded = padder.update(plaintext) + padder.finalize()

    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    ciphertext = encryptor.update(padded) + encryptor.finalize()

    return EncryptedTemplate(ciphertext=ciphertext, iv=iv)


def aes_decrypt(record: EncryptedTemplate, key: bytes) -> bytes:
    try:
        decryptor = Cipher(algorithms.AES(key), modes.CBC(record.iv)).decryptor()
        padded = decryptor.update(record.ciphertext) + decryptor.finalize()

        unpadder = padding.PKCS7(_BLOCK_BITS).unpadder()
        plaintext = unpadder.update(padded) + unpadder.finalize()
        return plaintext
    except ValueError as e:
        if "Invalid padding bytes" in str(e):
            raise ValueError(
                "Decryption failed: Invalid padding. This usually means the AES master key "
                "used for decryption is different from the key used for encryption. "
                "Ensure AES_MASTER_KEY_HEX environment variable is set to the same value "
                "used when the templates were originally encrypted."
            ) from e
        raise
