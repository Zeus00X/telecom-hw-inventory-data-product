import pandas as pd
from sqlalchemy import create_engine, text

def build_engine(server: str, database: str, username: str, password: str) :
    driver = "ODBC Driver 17 for SQL Server"
    url = (
        f"mssql+pyodbc://{username}:{password}@{server}:1433/{database}"
        f"?driver={driver.replace(' ', '+')}"
        f"&Encrypt=yes"
    )
    # fast_executemany acelera inserts masivos con pyodbc
    return create_engine(url, fast_executemany=True)

def truncate_and_load(
    engine,
    df: pd.DataFrame,
    schema: str,
    table: str,
    chunksize: int = 5000
) -> None:
    """
    Reemplazo total:
      1) TRUNCATE TABLE
      2) INSERT masivo con pandas.to_sql
    """
    # Normalizaciones mínimas
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
        chunksize=chunksize
    )
