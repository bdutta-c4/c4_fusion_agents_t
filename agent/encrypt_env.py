#!/usr/bin/env python3
"""
Encrypt .env → .env.enc using a passphrase.
Usage:  python encrypt_env.py --env .env --out .env.enc --passphrase "my secret"
"""
import argparse, base64, os, sys, secrets
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC   # PBKDF2-HMAC-SHA256
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet
import boto3
import json
import os


def derive_key(passphrase: str, salt: bytes, iterations: int = 100_000) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt,
                     iterations=iterations)
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))

def main(env: str, path_out: str, passphrase: str):
    secret_name = f"C-4analytics/fusion/{env}"
    aws_region = os.environ.get("SECRETS_AWS_REGION", "us-east-1")

    try:
        session = boto3.session.Session()
        client = session.client(service_name='secretsmanager', region_name=aws_region)
        secret_value = client.get_secret_value(SecretId=secret_name)
        env_secrets = json.loads(secret_value['SecretString'])
    except Exception as e:
        raise RuntimeError(f"Failed to obtain AWS variables for {env} environment in region {aws_region}: {e}") from e

    with open("envfile.txt","w") as f:
        for key, value in env_secrets.items():
            if isinstance(value, (str, int, float, bool)):
                if key!="SNOWFLAKE_PRIVATE_KEY":
                    envln = key+"="+value
                    f.write(envln + "\n")

    salt = secrets.token_bytes(16)                 # 128-bit salt
    key  = derive_key(passphrase, salt)
    data = open("envfile.txt", "rb").read()
    cipher = Fernet(key).encrypt(data)
    with open(path_out, "wb") as f:
        f.write(base64.b64encode(salt) + b"\n" + cipher)
    print(f"Encrypted {env} → {path_out}  (keep passphrase & salt safe!)")
    os.remove("envfile.txt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="dev")
    p.add_argument("--out", default=".env.enc")
    p.add_argument("--passphrase", required=True)
    a = p.parse_args()
    main(a.env, a.out, a.passphrase)