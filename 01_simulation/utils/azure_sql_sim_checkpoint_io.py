# 01_simulation/utils/azure_sql_sim_checkpoint_io.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import pandas as pd
from sqlalchemy import text


@dataclass
class Checkpoint:
    checkpoint_name: str = "default"
    last_year_week: Optional[str] = None          # 'YYYY_WW' o None
    last_year_week_key: Optional[int] = None      # YYYY*100+WW o None
    last_run_id: Optional[str] = None             # UUID string
    status: str = "OK"                            # OK / FAILED
    updated_at_utc: Optional[str] = None          # string UTC si decides poblarla


SIM_TABLES_CORE = [
    ("sim", "checkpoint_control"),
    ("sim", "inventory_ref_state"),
    ("sim", "orders_header_state"),
    ("sim", "orders_detail_state"),
]

SIM_TABLES_OPTIONAL = [
    ("sim", "transactions_fact"),
    ("sim", "inventory_weekly_snapshot"),
]


def reset_sim_tables(engine, include_optionals: bool = True) -> None:
    """
    Borra toda la información de las tablas de simulación.
    Usa TRUNCATE por rendimiento; requiere permisos.
    """
    tables = SIM_TABLES_CORE + (SIM_TABLES_OPTIONAL if include_optionals else [])
    with engine.begin() as conn:
        for schema, table in tables:
            conn.execute(text(f"TRUNCATE TABLE {schema}.{table};"))


def read_checkpoint(engine, checkpoint_name: str = "default") -> Checkpoint:
    """
    Lee el checkpoint actual. Si no existe registro, retorna Checkpoint vacío.
    """
    q = text("""
        SELECT TOP 1
            checkpoint_name,
            last_year_week,
            last_year_week_key,
            last_run_id,
            status
        FROM sim.checkpoint_control
        WHERE checkpoint_name = :checkpoint_name
        ORDER BY updated_at_utc DESC
    """)
    with engine.begin() as conn:
        rows = conn.execute(q, {"checkpoint_name": checkpoint_name}).fetchall()

    if not rows:
        return Checkpoint(checkpoint_name=checkpoint_name)

    r = rows[0]
    return Checkpoint(
        checkpoint_name=r[0],
        last_year_week=r[1],
        last_year_week_key=r[2],
        last_run_id=r[3],
        status=r[4] if r[4] else "OK",
    )


def upsert_checkpoint(
    engine,
    checkpoint: Checkpoint,
) -> None:
    """
    Upsert simple por checkpoint_name.
    """
    sql = text("""
        MERGE sim.checkpoint_control AS tgt
        USING (SELECT
            :checkpoint_name AS checkpoint_name,
            :last_year_week AS last_year_week,
            :last_year_week_key AS last_year_week_key,
            :last_run_id AS last_run_id,
            :status AS status
        ) AS src
        ON (tgt.checkpoint_name = src.checkpoint_name)
        WHEN MATCHED THEN
            UPDATE SET
                last_year_week = src.last_year_week,
                last_year_week_key = src.last_year_week_key,
                last_run_id = src.last_run_id,
                status = src.status,
                updated_at_utc = SYSUTCDATETIME()
        WHEN NOT MATCHED THEN
            INSERT (checkpoint_name, last_year_week, last_year_week_key, last_run_id, status, updated_at_utc)
            VALUES (src.checkpoint_name, src.last_year_week, src.last_year_week_key, src.last_run_id, src.status, SYSUTCDATETIME());
    """)
    with engine.begin() as conn:
        conn.execute(sql, {
            "checkpoint_name": checkpoint.checkpoint_name,
            "last_year_week": checkpoint.last_year_week,
            "last_year_week_key": checkpoint.last_year_week_key,
            "last_run_id": checkpoint.last_run_id,
            "status": checkpoint.status,
        })


def load_state_tables(engine) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga estado actual (sim.*_state) desde SQL a DataFrames.
    """
    df_inv = pd.read_sql("SELECT * FROM sim.inventory_ref_state", con=engine)
    df_hdr = pd.read_sql("SELECT * FROM sim.orders_header_state", con=engine)
    df_det = pd.read_sql("SELECT * FROM sim.orders_detail_state", con=engine)
    return df_inv, df_hdr, df_det


def save_state_tables(
    engine,
    df_inventory_ref_state: pd.DataFrame,
    df_orders_header_state: pd.DataFrame,
    df_orders_detail_state: pd.DataFrame,
) -> None:
    """
    Reemplaza estado completo (TRUNCATE + INSERT).
    Se usa porque el estado es "la verdad" del checkpoint, no un histórico.
    """
    # Normalizaciones mínimas para evitar problemas de tipos
    df_inventory_ref_state = df_inventory_ref_state.copy()
    df_orders_header_state = df_orders_header_state.copy()
    df_orders_detail_state = df_orders_detail_state.copy()

    # Es buena práctica asegurar columnas y tipos críticos aquí si lo necesitas

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE sim.inventory_ref_state;"))
        conn.execute(text("TRUNCATE TABLE sim.orders_header_state;"))
        conn.execute(text("TRUNCATE TABLE sim.orders_detail_state;"))

    df_inventory_ref_state.to_sql(
        name="inventory_ref_state", con=engine, schema="sim",
        if_exists="append", index=False, chunksize=5000
    )
    df_orders_header_state.to_sql(
        name="orders_header_state", con=engine, schema="sim",
        if_exists="append", index=False, chunksize=5000
    )
    df_orders_detail_state.to_sql(
        name="orders_detail_state", con=engine, schema="sim",
        if_exists="append", index=False, chunksize=5000
    )


def append_transactions_fact(engine, df_tx: pd.DataFrame) -> None:
    """
    Inserta (append) transacciones históricas.
    Importante: idealmente la tabla tiene PK por id_transaccion para evitar duplicados.
    """
    if df_tx is None or df_tx.empty:
        return
    df_tx.to_sql(
        name="transactions_fact", con=engine, schema="sim",
        if_exists="append", index=False, chunksize=5000
    )


def append_inventory_weekly_snapshot(engine, df_snap: pd.DataFrame) -> None:
    """
    Inserta (append) snapshot semanal histórico.
    Ideal: PK compuesta (run_id, year_week, id_referencia) o similar.
    """
    if df_snap is None or df_snap.empty:
        return
    df_snap.to_sql(
        name="inventory_weekly_snapshot", con=engine, schema="sim",
        if_exists="append", index=False, chunksize=5000
    )
