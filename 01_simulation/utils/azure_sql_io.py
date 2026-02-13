# 01_simulation/utils/azure_sql_io.py

import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()


def build_engine(server: str, database: str, username: str, password: str):
    """
    Crea engine hacia Azure SQL de forma robusta (soporta caracteres especiales en password como '#').
    """
    driver = "ODBC Driver 17 for SQL Server"

    connection_url = URL.create(
        "mssql+pyodbc",
        username=username,
        password=password,
        host=server,   # ej: sqltelecomhwdevsa01.database.windows.net
        port=1433,
        database=database,
        query={
            "driver": driver,
            "Encrypt": "yes",
            "TrustServerCertificate": "no",
        },
    )

    return create_engine(
        connection_url,
        fast_executemany=True,
        pool_pre_ping=True,
        connect_args={"timeout": 60},
    )


def build_engine_from_env():
    """
    Lee credenciales desde .env / variables de entorno:
      AZ_SQL_SERVER, AZ_SQL_DB, AZ_SQL_USER, AZ_SQL_PASSWORD
    """
    server = os.getenv("AZ_SQL_SERVER")
    database = os.getenv("AZ_SQL_DB")
    username = os.getenv("AZ_SQL_USER")
    password = os.getenv("AZ_SQL_PASSWORD")

    if not all([server, database, username, password]):
        raise ValueError(
            "Faltan variables de entorno de Azure SQL: "
            "AZ_SQL_SERVER, AZ_SQL_DB, AZ_SQL_USER, AZ_SQL_PASSWORD"
        )

    return build_engine(server, database, username, password)


def read_table(engine, schema: str, table: str) -> pd.DataFrame:
    """
    Lee una tabla completa desde Azure SQL.
    """
    return pd.read_sql(f"SELECT * FROM {schema}.{table}", con=engine)


def truncate_and_load(
    engine,
    df: pd.DataFrame,
    schema: str,
    table: str,
    chunksize: int = 5000,
) -> None:
    """
    Reemplazo total:
      1) TRUNCATE TABLE
      2) INSERT masivo con pandas.to_sql

    Nota: Esta función asume que la tabla ya existe y tiene el esquema esperado.
    """
    df = df.copy()

    # BIT en SQL: 0/1
    if "fue_contaminada" in df.columns:
        df["fue_contaminada"] = df["fue_contaminada"].astype(int)

    # DATETIME2: asegurar datetime sin tz
    if "fc_instalacion" in df.columns:
        df["fc_instalacion"] = pd.to_datetime(df["fc_instalacion"]).dt.tz_localize(None)

    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {schema}.{table};"))

    df.to_sql(
        name=table,
        con=engine,
        schema=schema,
        if_exists="append",
        index=False,
        chunksize=chunksize,
    )
