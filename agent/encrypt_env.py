#!/usr/bin/env python3
"""
Encrypt .env → .env.enc using a passphrase.
Usage:  python encrypt_env.py --env .env --out .env.enc --passphrase "my secret"
"""
import argparse, base64, os, sys, secrets
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC   # PBKDF2-HMAC-SHA256
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet

def derive_key(passphrase: str, salt: bytes, iterations: int = 100_000) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt,
                     iterations=iterations)
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))

def main(path_in: str, path_out: str, passphrase: str):
    salt = secrets.token_bytes(16)                 # 128-bit salt
    key  = derive_key(passphrase, salt)
    data = open(path_in, "rb").read()
    cipher = Fernet(key).encrypt(data)
    with open(path_out, "wb") as f:
        f.write(base64.b64encode(salt) + b"\n" + cipher)
    print(f"Encrypted {path_in} → {path_out}  (keep passphrase & salt safe!)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default=".env")
    p.add_argument("--out", default=".env.enc")
    p.add_argument("--passphrase", required=True)
    a = p.parse_args()
    if not os.path.exists(a.env):
        sys.exit(f"{a.env} not found")
    main(a.env, a.out, a.passphrase)