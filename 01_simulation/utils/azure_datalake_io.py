# 01_simulation/utils/azure_datalake_io.py

from __future__ import annotations

import os
import io
import pandas as pd
from azure.storage.blob import BlobServiceClient


def _get_blob_service_client() -> BlobServiceClient:
    """
    Autenticación por connection string.
    Requiere env var:
      AZ_STORAGE_CONNECTION_STRING
    """
    conn_str = os.getenv("AZ_STORAGE_CONNECTION_STRING") or os.getenv("AZ_BLOB_CONN_STR")
    if not conn_str:
        raise ValueError("Falta variable de entorno: AZ_STORAGE_CONNECTION_STRING")
    return BlobServiceClient.from_connection_string(conn_str)


def upload_df_as_csv(
    df: pd.DataFrame,
    blob_path: str,
    container_name: str | None = None,
    overwrite: bool = True,
    encoding: str = "utf-8",
) -> None:
    """
    Sube un DataFrame como CSV a Azure Blob / ADLS Gen2 (vía Blob API).
    Requiere:
      AZ_STORAGE_CONTAINER (si no pasas container_name)
    """
    if container_name is None:
        container_name = os.getenv("AZ_STORAGE_CONTAINER") or os.getenv("AZ_BLOB_CONTAINER")
    if not container_name:
        raise ValueError("Falta AZ_STORAGE_CONTAINER o container_name")

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    data = buffer.getvalue().encode(encoding)

    bsc = _get_blob_service_client()
    blob_client = bsc.get_blob_client(container=container_name, blob=blob_path)
    blob_client.upload_blob(data, overwrite=overwrite)
