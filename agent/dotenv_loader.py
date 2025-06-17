"""
Decrypt .env.enc, export vars into os.environ, return dict.
Usage (CLI):  python dotenv_loader.py --enc .env.enc [--passphrase "secret"]
Usage (import):   from dotenv_loader import load_encrypted_dotenv; load_encrypted_dotenv()
"""
import argparse, base64, os, sys
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet
from dotenv import dotenv_values
from io import StringIO

def _derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32,
                     salt=salt, iterations=100_000)
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))

def load_encrypted_dotenv(path: str = ".env.enc",
                          passphrase: str | None = None) -> dict:
    passphrase = passphrase or os.getenv("DOTENV_PASSPHRASE")
    if not passphrase:
        raise RuntimeError("No passphrase supplied (env DOTENV_PASSPHRASE missing)")
    with open(path, "rb") as f:
        salt_b64, cipher = f.read().split(b"\n", 1)
    key = _derive_key(passphrase, base64.b64decode(salt_b64))
    plaintext = Fernet(key).decrypt(cipher)
    plaintext_str = plaintext.decode()          # bytes ➜ str
    env_stream    = StringIO(plaintext_str)     # make it file-like
    env_dict      = dotenv_values(stream=env_stream)  # ← no AttributeError
    os.environ.update(env_dict)
    return env_dict

if __name__ == "__main__":                         # CLI helper
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default=".env.enc")
    ap.add_argument("--passphrase")
    args = ap.parse_args()
    envs = load_encrypted_dotenv(args.enc, args.passphrase)
    print(f"Loaded {len(envs)} variables into environment")