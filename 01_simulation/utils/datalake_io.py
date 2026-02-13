import os
import pandas as pd
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

def _get_blob_service_client() -> BlobServiceClient:
    conn_str = os.getenv("AZ_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise ValueError("Falta AZ_STORAGE_CONNECTION_STRING en .env / variables de entorno")
    return BlobServiceClient.from_connection_string(conn_str)

def upload_df_as_csv(
    df: pd.DataFrame,
    blob_path: str,
    container: str | None = None,
    encoding: str = "utf-8",
) -> None:
    """
    Sube un DataFrame como CSV a Azure Blob Storage (contenedor tipo 'datalake').

    blob_path ejemplo:
      'transacciones/year_week=2026_21/df_transacciones_2026_21.csv'
    """
    container_name = container or os.getenv("AZ_STORAGE_CONTAINER", "datalake")
    bsc = _get_blob_service_client()
    blob_client = bsc.get_blob_client(container=container_name, blob=blob_path)

    csv_bytes = df.to_csv(index=False).encode(encoding)
    blob_client.upload_blob(csv_bytes, overwrite=True)
