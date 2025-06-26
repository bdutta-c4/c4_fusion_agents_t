import boto3
import json
import os
import logging

logger = logging.getLogger("fusion")

def get_environment():
    """Detect the current environment"""
    return os.environ.get("APP_ENVIRONMENT", "local")


def is_production():
    """Check if running in production"""
    return get_environment().lower() == "prod"


def is_development():
    """Check if running in development"""
    return get_environment().lower() == "dev"



def load_aws_variables():
    """
    Load secrets from AWS Secrets Manager and inject them directly into environment variables.

    Returns:
        dict: The loaded secrets for reference, though they are already in os.environ
    """
    env = get_environment()
    secret_name = f"C-4analytics/fusion/{env}"
    logger.info(f"Loading variables for environment: {env}")
    aws_region = os.environ.get("SECRETS_AWS_REGION", "us-east-1")

    try:
        session = boto3.session.Session()
        client = session.client(service_name='secretsmanager', region_name=aws_region)
        secret_value = client.get_secret_value(SecretId=secret_name)
        secrets = json.loads(secret_value['SecretString'])
        logger.info("Successfully obtained variables from AWS Secrets Manager")
    except Exception as e:
        logger.warning(f"Failed to obtain variables from AWS Secrets Manager in region {aws_region}: {e}")
        if env == "local":
            logger.info("In local environment - using existing environment variables")
            secrets = {}
        else:
            logger.error(f"Failed to obtain variables for {env} environment in region {aws_region}: {e}")
            raise RuntimeError(f"Failed to obtain AWS variables for {env} environment in region {aws_region}: {e}") from e

    # Inject all secrets into environment variables
    for key, value in secrets.items():
        if isinstance(value, (str, int, float, bool)):
            # Convert all values to strings for environment variables
            os.environ[key] = str(value)
        else:
            logger.warning(f"Skipping environment injection for non-string variables: {key}")

    logger.info(f"Loaded {len(secrets)} AWS secrets into environment variables")

    return secrets

def load_encrypted_variables():
    # secrets = load_aws_variables()
    # if secrets:
    #    return True
    # else:
    #    return False
    from dotenv_loader import load_encrypted_dotenv
    from pathlib import Path

    passphrase=os.getenv("DOTENV_PASSPHRASE")
    BASE_DIR = Path(__file__).resolve().parent   
    #load_encrypted_dotenv(".env.enc",passphrase)  
    env = os.getenv("APP_ENVIRONMENT", "dev")
    env_path = f".env.enc.{env}"
    load_encrypted_dotenv(BASE_DIR / env_path,passphrase)
    return True