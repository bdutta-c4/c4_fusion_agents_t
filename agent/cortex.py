'''
File: cortex.py

Cortex / Snowflake utils for Fusion AI

Copyright 2025 C-4 Analytics, LLC
'''

import logging
import sys
from dataclasses import dataclass
from functools import wraps
from time import sleep
from typing import Any, Callable, Dict

import pandas as pd
import requests
import snowflake.connector
import snowflake.connector.network
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from snowflake.core import Root

logger = logging.getLogger("fusion")


@dataclass
class SnowflakeConfig:
    """Dataclass containing Snowflake connection details"""
    account: str
    user: str
    warehouse: str
    role: str
    host: str
    database: str
    schema: str
    private_key_file: str | None = None
    private_key: str | None = None
    private_key_passphrase: str | None = None


@dataclass
class CortexConfig:
    """Dataclass containing Cortex Analyst configuration details"""
    database: str
    schema: str
    stage: str
    semantic_model_filename: str


def get_snowflake_connection(config: SnowflakeConfig) -> snowflake.connector.SnowflakeConnection:
    """Connect to snowflake via provided credentials"""
    if config.private_key_file is not None:
        return snowflake.connector.connect(
            account=config.account,
            user=config.user,
            port=443,
            warehouse=config.warehouse,
            database=config.database,
            schema=config.schema,
            role=config.role,
            private_key_file=config.private_key_file
        )
    else:
        try:
            p_key = serialization.load_pem_private_key(
                config.private_key,
                password=config.private_key_passphrase,
                backend=default_backend()
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load private key: {e}")

        try:
            pkb = p_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        except Exception as e:
            raise RuntimeError(f"Failed to serialize private key: {e}")

        return snowflake.connector.connect(
            user=config.user,
            private_key=pkb,
            account=config.account,
            host=config.host,
            port=443,
            warehouse=config.warehouse,
            role=config.role,
            database=config.database,
            schema=config.schema
        )


def handle_errors(max_retries: int = 3):
    def decorator_retry(func: Callable) -> Callable:
        @wraps(func)
        def wrapper_retry(self, *args: Any, **kwargs: Any) -> Any:
            try:
                for attempt in range(max_retries):
                    try:
                        return func(self, *args, **kwargs)
                    except (snowflake.connector.network.ReauthenticationRequest, snowflake.connector.errors.DatabaseError):
                        self.close()
                        self.open()
                raise RuntimeError("Failed to invoke Snowflake API")
            finally:
                # this is unbelievable - this is an issue with snowflake
                # setting `sys.tracebacklimit = 0` for everything
                # so we must clean up their mess
                sys.tracebacklimit = None
        return wrapper_retry

    # Handle both @retry and @retry() cases
    if callable(max_retries):
        # If no arguments, max_retries is the function itself
        original_func = max_retries
        max_retries = 3  # default value
        return decorator_retry(original_func)

    return decorator_retry


class CortexHandler:

    def __init__(self, snowflake_config: SnowflakeConfig, cortex_config: CortexConfig):
        self.snowflake_config: SnowflakeConfig = snowflake_config
        self.cortex_config: CortexConfig = cortex_config
        self.snowflake_connection: snowflake.connector.SnowflakeConnection = None
        self.snowflake_root: Root = None

    def open(self):
        """Open the Snowflake connection"""
        if self.snowflake_connection is None:
            self.snowflake_connection = get_snowflake_connection(
                self.snowflake_config)
            self.snowflake_root = Root(self.snowflake_connection)

    def close(self):
        if self.snowflake_connection is not None:
            self.snowflake_connection.close()
            self.snowflake_connection = None
            self.snowflake_root = None

    @handle_errors
    def send_message(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Calls the Cortex REST API and returns the response."""

        for attempt in range(max_retries):
            cfg = self.cortex_config
            request_body = {
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                "semantic_model_file": f"@{cfg.database}.{cfg.schema}.{cfg.stage}/{cfg.semantic_model_filename}",
            }
            resp = requests.post(
                url=f"https://{self.snowflake_config.host}/api/v2/cortex/analyst/message",
                json=request_body,
                headers={
                    "Authorization": f'Snowflake Token="{self.snowflake_connection.rest.token}"',
                    "Content-Type": "application/json",
                },
            )
            request_id = resp.headers.get("X-Snowflake-Request-Id")

            match resp.status_code:
                case 503 | 504:
                    # service unavailable, back off and retry...
                    sleep_time = 2 ** attempt
                    logger.warning(
                        f"Cortex API returned status code {resp.status_code}. Retrying in {sleep_time} seconds...")
                    sleep(sleep_time)
                    continue
                case code if code < 400:
                    # success
                    pass
                case _:
                    raise Exception(
                        f"Failed request (id: {request_id}) with status {resp.status_code}: {resp.text}")

            resp_data = resp.json()

            match resp_data:
                case {"error_code": "390112"} | {"code": "390112"}:
                    # session expired, need to reconnect and retry
                    logger.warning("Cortex session expired, retrying...")
                    self.close()
                    self.open()
                    continue
                case {"error_code": "390111"} | {"code": "390111"}:
                    # session no longer exists
                    logger.warning(
                        "Cortex session no longer exists, retrying...")
                    self.close()
                    self.open()
                    continue
                case _:
                    return {**resp_data, "request_id": request_id}
        else:
            raise RuntimeError("Failed to invoke Cortex API")

    @handle_errors
    def resolve_client_details(self, query: str) -> str:
        """Resolve the client details from the given JIRA ID or client name"""

        logger.info(f"Resolve client details for::: {query}")

        response = (
            self.snowflake_root.databases[self.snowflake_config.database]
            .schemas[self.snowflake_config.schema]
            .cortex_search_services["fusion_data_client_name_search_service"]
            .search(
                query,
                ["JIRA_ID"],
                limit=1
            )
        )

        res = response.results

        if not res:
            return f"No client with JIRA ID: {jira_id}"

        jira_id = res[0]["JIRA_ID"]
        sql = f"""SELECT JIRA_ID, DEALERSHIP_NAME, DEALERSHIP_GROUP, BRAND, CITY, STATE, ZIP, OEM_PROGRAM, OEM_NAME FROM {self.snowflake_config.database}.{self.snowflake_config.schema}.CLIENT_DETAILS WHERE JIRA_ID = '{jira_id}';"""
        df = pd.read_sql(sql, self.snowflake_connection)
        if df.empty:
            return f"No client with JIRA ID: {jira_id}"
        return df.iloc[0].to_dict()

    @handle_errors
    def match_clients(self, query: str, limit: int = 5) -> pd.DataFrame:
        """Match the client details from the given JIRA ID or client name and return the K closest matches as a DataFrame"""

        logger.info(f"Resolve client details for::: {query}")

        response = (
            self.snowflake_root.databases[self.snowflake_config.database]
            .schemas[self.snowflake_config.schema]
            .cortex_search_services["fusion_data_client_name_search_service"]
            .search(
                query,
                ["JIRA_ID"],
                limit=limit
            )
        )

        res = response.results

        jira_ids = [f"'{x['JIRA_ID']}'" for x in res]
        jira_ids_fmt = ",".join(jira_ids)
        sql = f"""SELECT JIRA_ID, DEALERSHIP_NAME, DEALERSHIP_GROUP, BRAND, CITY, STATE, ZIP, OEM_PROGRAM, OEM_NAME FROM {self.snowflake_config.database}.{self.snowflake_config.schema}.CLIENT_DETAILS WHERE JIRA_ID IN ({jira_ids_fmt});"""
        return pd.read_sql(sql, self.snowflake_connection)
