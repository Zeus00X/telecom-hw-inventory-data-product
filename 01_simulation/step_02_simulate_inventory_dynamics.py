#%%
# step_02_simulate_inventory_dynamics.py
# -------------------------------------------------------------------
# Dinámica v2 (Data Engineer):
# - Lee insumos desde Azure SQL (stg.*)
# - Modelo A (consistencia fuerte) con checkpoint en Azure SQL (sim.*)
# - Permite RESET (borrar sim.* y reiniciar)
# - Permite ejecutar rango de semanas o semana única
# - Genera CSV semanal de transacciones y lo sube al Data Lake (overwrite)
# - Persiste estado (inventario + órdenes) como checkpoint en Azure SQL
# -------------------------------------------------------------------

import os
import uuid
import random
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from utils.azure_sql_io import (
    build_engine_from_env,
    read_table,
    truncate_and_load,
)

from utils.azure_datalake_io import upload_df_as_csv


#%% ===============================================================
# 1) PARÁMETROS DE EJECUCIÓN
# ===============================================================

# --- RESET: borra sim.* y reinicia checkpoint
RESET = True

# --- Rango a ejecutar (si quieres semana única, pon MIN=MAX)
MIN_WEEK_SIM = "2025_01"
MAX_WEEK_SIM = "2025_52"

# --- Modelo A: consistencia fuerte
# Si hay checkpoint y pides empezar después de la siguiente semana esperada:
# - auto-fill: ajusta MIN_WEEK_SIM automáticamente para rellenar huecos
# - reject: levanta error
MODEL_A_MODE = "auto_fill"  # "auto_fill" | "reject"

# --- Data Lake: carpeta base donde quedan los CSV semanales
DATALAKE_BASE_PATH = "telecom-hw/inventory_simulation/transacciones"

# --- Persistencia opcional (si existen tablas)
WRITE_OPTIONAL_FACTS = False  # sim.transactions_fact, sim.inventory_weekly_snapshot

# --- Exportación local opcional (para debug/portafolio)
EXPORT_EXCEL = False
CARPETA_SALIDA = "01_simulation/output"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

RUTA_TRANSACCIONES_OUT = f"{CARPETA_SALIDA}/02_transacciones_inventario.xlsx"
RUTA_INVENTARIO_SEMANAL_OUT = f"{CARPETA_SALIDA}/02_inventario_semanal.xlsx"
RUTA_ORDENES_RESUMEN_OUT = f"{CARPETA_SALIDA}/02_ordenes_resumen.xlsx"
RUTA_ORDENES_DETALLE_OUT = f"{CARPETA_SALIDA}/02_ordenes_detalle.xlsx"


#%% ===============================================================
# 2) UTILIDADES DE TIEMPO / SEMANAS
# ===============================================================

def random_datetime_in_year_week(year_week: str, seed_key: str | None = None) -> str:
    """Retorna una fecha aleatoria dentro de la semana ISO 'YYYY_WW' (string)."""
    if not isinstance(year_week, str) or "_" not in year_week:
        raise ValueError("year_week debe tener formato 'YYYY_WW' (ej: '2025_01').")

    y_str, w_str = year_week.split("_")
    year = int(y_str)
    week = int(w_str)

    monday = pd.to_datetime(f"{year}-W{week:02d}-1", format="%G-W%V-%u").to_pydatetime()

    rnd = random.Random(f"{year_week}|{seed_key}") if seed_key is not None else random
    day_offset = rnd.randint(0, 6)
    hour = rnd.randint(0, 23)
    minute = rnd.randint(0, 59)
    second = rnd.randint(0, 59)

    dt = monday + pd.Timedelta(days=day_offset)
    dt = dt.replace(hour=hour, minute=minute, second=second, microsecond=0)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def year_week_to_key(year_week: str) -> int:
    """Convierte 'YYYY_WW' a entero YYYY*100+WW para ordenar."""
    year, week = year_week.split("_")
    return int(year) * 100 + int(week)


def key_to_year_week(key: int) -> str:
    """Convierte entero YYYY*100+WW a 'YYYY_WW'."""
    year = key // 100
    week = key % 100
    return f"{year}_{week:02d}"


def _iso_weeks_in_year(year: int) -> int:
    """Retorna el número de semanas ISO del año (52 o 53)."""
    return pd.Timestamp(year=year, month=12, day=28).isocalendar().week


def next_year_week(yw: str) -> str:
    """Siguiente semana ISO de 'YYYY_WW'."""
    y, w = yw.split("_")
    year = int(y)
    week = int(w)
    weeks_in_year = _iso_weeks_in_year(year)

    week += 1
    if week > weeks_in_year:
        year += 1
        week = 1
    return f"{year}_{week:02d}"


def build_horizon_year_weeks(min_week_sim: str, max_week_sim: str) -> list[str]:
    """Construye horizonte continuo ISO entre min y max (incluyentes)."""
    if year_week_to_key(min_week_sim) > year_week_to_key(max_week_sim):
        raise ValueError("MIN_WEEK_SIM debe ser <= MAX_WEEK_SIM")

    y0, w0 = min_week_sim.split("_")
    y1, w1 = max_week_sim.split("_")

    year = int(y0)
    week = int(w0)
    end_year = int(y1)
    end_week = int(w1)

    horizon = []
    while True:
        horizon.append(f"{year}_{week:02d}")
        if year == end_year and week == end_week:
            break

        weeks_in_year = _iso_weeks_in_year(year)
        week += 1
        if week > weeks_in_year:
            year += 1
            week = 1
    return horizon


#%% ===============================================================
# 3) HELPERS SQL (checkpoint + reset)
# ===============================================================

def get_checkpoint(engine) -> dict | None:
    """Lee el checkpoint actual (si existe)."""
    q = """
        SELECT TOP 1
            checkpoint_id,
            last_year_week_processed,
            updated_at,
            status
        FROM sim.checkpoint_control
        ORDER BY updated_at DESC;
    """
    df = pd.read_sql(q, con=engine)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def set_checkpoint(engine, last_year_week_processed: str, status: str = "OK") -> None:
    """Setea checkpoint como una sola fila (patrón simple para portafolio)."""
    checkpoint_id = str(uuid.uuid4())
    updated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE sim.checkpoint_control;"))
        conn.execute(
            text("""
                INSERT INTO sim.checkpoint_control (
                    checkpoint_id, last_year_week_processed, updated_at, status
                )
                VALUES (:checkpoint_id, :last_year_week_processed, :updated_at, :status);
            """),
            {
                "checkpoint_id": checkpoint_id,
                "last_year_week_processed": last_year_week_processed,
                "updated_at": updated_at,
                "status": status,
            },
        )


def reset_sim_schema(engine) -> None:
    """Borra estado sim.* (RESET)."""
    with engine.begin() as conn:
        # Orden seguro: detalle -> cabecera -> inventario -> checkpoint
        conn.execute(text("TRUNCATE TABLE sim.orders_detail_state;"))
        conn.execute(text("TRUNCATE TABLE sim.orders_header_state;"))
        conn.execute(text("TRUNCATE TABLE sim.inventory_ref_state;"))
        conn.execute(text("TRUNCATE TABLE sim.checkpoint_control;"))

        # Opcionales (si existen)
        if WRITE_OPTIONAL_FACTS:
            try:
                conn.execute(text("TRUNCATE TABLE sim.transactions_fact;"))
            except Exception:
                pass
            try:
                conn.execute(text("TRUNCATE TABLE sim.inventory_weekly_snapshot;"))
            except Exception:
                pass


def get_insertable_columns_sqlserver(engine, schema: str, table: str) -> list[str]:
    """
    Retorna las columnas insertables de una tabla SQL Server:
    - Excluye computed columns
    - Excluye identity columns
    - Excluye rowversion/timestamp
    """
    sql = text("""
        SELECT
            c.name AS column_name,
            c.is_identity,
            CASE WHEN cc.column_id IS NULL THEN 0 ELSE 1 END AS is_computed,
            ty.name AS type_name
        FROM sys.tables t
        JOIN sys.schemas s ON s.schema_id = t.schema_id
        JOIN sys.columns c ON c.object_id = t.object_id
        JOIN sys.types ty ON ty.user_type_id = c.user_type_id
        LEFT JOIN sys.computed_columns cc
               ON cc.object_id = c.object_id AND cc.column_id = c.column_id
        WHERE s.name = :schema
          AND t.name = :table
        ORDER BY c.column_id;
    """)
    meta = pd.read_sql(sql, con=engine, params={"schema": schema, "table": table})

    # rowversion/timestamp: no se inserta manualmente
    mask_insertable = (
        (meta["is_identity"] == 0) &
        (meta["is_computed"] == 0) &
        (~meta["type_name"].isin(["timestamp", "rowversion"]))
    )

    cols = meta.loc[mask_insertable, "column_name"].tolist()
    return cols


def sanitize_df_for_sqlserver_table(engine, df: pd.DataFrame, schema: str, table: str) -> pd.DataFrame:
    """
    Deja el DF listo para INSERT:
    - Toma solo columnas insertables existentes en el DF
    - Elimina columnas "problemáticas" (computed/identity/etc.)
    """
    if df is None or df.empty:
        return df

    insertable_cols = get_insertable_columns_sqlserver(engine, schema, table)

    # Intersección (respeta el orden de la tabla)
    cols_to_keep = [c for c in insertable_cols if c in df.columns]

    # Si no hay intersección, devuelve DF tal cual (pero esto sería raro)
    if not cols_to_keep:
        return df

    return df[cols_to_keep].copy()



def persist_state(engine, df_inventario_ref, df_ordenes, df_ordenes_detalle) -> None:
    """Persistencia full-replace de estado (checkpoint) en sim.*."""

    # --- IMPORTANTE: eliminar columnas no insertables (computed/identity)
    df_inventario_ref_s = sanitize_df_for_sqlserver_table(engine, df_inventario_ref, "sim", "inventory_ref_state")
    df_ordenes_s        = sanitize_df_for_sqlserver_table(engine, df_ordenes,        "sim", "orders_header_state")
    df_det_s            = sanitize_df_for_sqlserver_table(engine, df_ordenes_detalle,"sim", "orders_detail_state")

    truncate_and_load(engine, df_inventario_ref_s, "sim", "inventory_ref_state")
    truncate_and_load(engine, df_ordenes_s, "sim", "orders_header_state")
    truncate_and_load(engine, df_det_s, "sim", "orders_detail_state")


def append_optional_facts(engine, df_tx_wk: pd.DataFrame, df_inv_snap_wk: pd.DataFrame) -> None:
    """Append a tablas opcionales si existen."""
    if not WRITE_OPTIONAL_FACTS:
        return

    with engine.begin() as conn:
        # transactions_fact (append)
        try:
            df_tx_wk.to_sql(
                name="transactions_fact",
                con=engine,
                schema="sim",
                if_exists="append",
                index=False,
                chunksize=5000,
            )
        except Exception:
            pass

        # inventory_weekly_snapshot (append)
        try:
            df_inv_snap_wk.to_sql(
                name="inventory_weekly_snapshot",
                con=engine,
                schema="sim",
                if_exists="append",
                index=False,
                chunksize=5000,
            )
        except Exception:
            pass


#%% ===============================================================
# 4) CONEXIÓN + LECTURA STG
# ===============================================================

engine = build_engine_from_env()

# RESET si aplica
if RESET:
    print("[RESET] Borrando sim.* y reiniciando checkpoint...")
    reset_sim_schema(engine)

# Insumos siempre desde stg (fuente de verdad)
df_sitios_real = read_table(engine, "stg", "sitios_x_configuracion_real")
df_pedidos = read_table(engine, "stg", "pedidos_desde_proyeccion")

# Normalización mínima
df_sitios_real["year_week"] = df_sitios_real["year_week"].astype(str)
df_pedidos["year_week_pedido"] = df_pedidos["year_week_pedido"].astype(str)
df_pedidos["year_week_disponible"] = df_pedidos["year_week_disponible"].astype(str)


#%% ===============================================================
# 5) PREPARACIÓN: LLEGADAS + ÓRDENES (estructura base)
# ===============================================================

# LLEGADAS (input para transacciones tipo LLEGADA)
df_llegadas = (
    df_pedidos
    .groupby(["year_week_disponible", "id_referencia", "referencia", "unidad"], as_index=False)
    .agg(cantidad_llegada=("cantidad_pedido", "sum"))
).rename(columns={"year_week_disponible": "year_week"})

# DEMANDA: creación de órdenes por sitio y semana, desde consumo real (cantidad_real)
df_ordenes_detalle_base = (
    df_sitios_real
    .groupby(
        ["id_sitio", "nombre_sitio", "region", "year_week", "id_referencia", "referencia", "unidad"],
        as_index=False
    )
    .agg(cantidad_solicitada=("cantidad_real", "sum"))
)

df_ordenes_detalle_base["id_orden"] = (
    df_ordenes_detalle_base["id_sitio"].astype(str) + "_" + df_ordenes_detalle_base["year_week"].astype(str)
)

# Campos operativos detalle (estado)
df_ordenes_detalle_base["cantidad_asignada"] = 0.0
df_ordenes_detalle_base["cantidad_pendiente"] = df_ordenes_detalle_base["cantidad_solicitada"].astype(float)
df_ordenes_detalle_base["estado_item"] = "PENDIENTE"
df_ordenes_detalle_base["week_ultima_reserva"] = pd.NA

# Cabecera
df_ordenes_base = (
    df_ordenes_detalle_base
    .groupby(["id_orden", "id_sitio", "nombre_sitio", "region", "year_week"], as_index=False)
    .agg(
        total_referencias=("id_referencia", "nunique"),
        total_solicitado=("cantidad_solicitada", "sum"),
        total_asignado=("cantidad_asignada", "sum"),
        total_pendiente=("cantidad_pendiente", "sum"),
    )
).rename(columns={"year_week": "year_week_necesidad"})

df_ordenes_base["estado_orden"] = "PENDIENTE"
df_ordenes_base["week_primera_reserva"] = pd.NA
df_ordenes_base["week_completado"] = pd.NA
df_ordenes_base["week_consumido"] = pd.NA
df_ordenes_base["fecha_proceso"] = pd.NA


#%% ===============================================================
# 6) MODELO A: RESOLVER RANGO A EJECUTAR SEGÚN CHECKPOINT
# ===============================================================

checkpoint = get_checkpoint(engine)

requested_min = MIN_WEEK_SIM
requested_max = MAX_WEEK_SIM

if checkpoint and checkpoint.get("last_year_week_processed"):
    last_processed = str(checkpoint["last_year_week_processed"])
    expected_next = next_year_week(last_processed)

    if year_week_to_key(requested_min) > year_week_to_key(expected_next):
        msg = (
            f"[Modelo A] Inconsistencia: checkpoint={last_processed} -> "
            f"esperado iniciar en {expected_next}, pero pediste {requested_min}."
        )
        if MODEL_A_MODE == "reject":
            raise ValueError(msg + " (MODE=reject)")
        else:
            print(msg + " (MODE=auto_fill) Ajustando MIN_WEEK_SIM automáticamente.")
            requested_min = expected_next

    if year_week_to_key(requested_max) <= year_week_to_key(last_processed):
        print(
            f"[Modelo A] Nada por ejecutar: requested_max={requested_max} "
            f"<= checkpoint={last_processed}."
        )
        # Aun así puedes querer regenerar CSV, pero por consistencia no re-procesamos.
        # Salimos limpio.
        raise SystemExit(0)

horizon_weeks = build_horizon_year_weeks(requested_min, requested_max)
print(f"[OK] Horizonte a ejecutar: {requested_min} -> {requested_max} | semanas={len(horizon_weeks)}")


#%% ===============================================================
# 7) CARGA/INICIALIZACIÓN DEL ESTADO (RESET vs CHECKPOINT)
# ===============================================================

def init_inventory_state_from_refs(df_llegadas_: pd.DataFrame, df_det_: pd.DataFrame) -> pd.DataFrame:
    refs_from_llegadas = df_llegadas_[["id_referencia", "referencia", "unidad"]].drop_duplicates()
    refs_from_demanda = df_det_[["id_referencia", "referencia", "unidad"]].drop_duplicates()

    df_refs = (
        pd.concat([refs_from_llegadas, refs_from_demanda], ignore_index=True)
        .drop_duplicates(subset=["id_referencia"])
    )

    df_inv = df_refs.copy()
    df_inv["on_hand"] = 0.0
    df_inv["reserved"] = 0.0
    df_inv["available"] = 0.0
    df_inv["year_week_corte"] = pd.NA
    df_inv["fecha_proceso"] = pd.NA
    return df_inv


if (checkpoint is None) or RESET:
    print("[INIT] No hay checkpoint (o RESET). Inicializando estado desde cero...")

    df_inventario_ref = init_inventory_state_from_refs(df_llegadas, df_ordenes_detalle_base)
    df_ordenes = df_ordenes_base.copy()
    df_ordenes_detalle = df_ordenes_detalle_base.copy()

    # Persistimos estado inicial (antes de procesar semanas)
    persist_state(engine, df_inventario_ref, df_ordenes, df_ordenes_detalle)

else:
    print("[RESUME] Cargando estado desde checkpoint en sim.* ...")
    df_inventario_ref = read_table(engine, "sim", "inventory_ref_state")
    df_ordenes = read_table(engine, "sim", "orders_header_state")
    df_ordenes_detalle = read_table(engine, "sim", "orders_detail_state")

    # Asegurar tipos mínimos
    df_ordenes["year_week_necesidad"] = df_ordenes["year_week_necesidad"].astype(str)
    df_ordenes_detalle["year_week"] = df_ordenes_detalle["year_week"].astype(str)

    # (Importante) Si el estado guardado quedó con year_week_corte viejo, no pasa nada:
    # se irá actualizando al final de cada semana procesada.


#%% ===============================================================
# 8) LÓGICA DE NEGOCIO: TRANSACCIONES E INVENTARIO
# ===============================================================

def add_transaction(
    year_week: str,
    tipo_movimiento: str,
    id_referencia: int | str,
    referencia: str,
    unidad: str,
    cantidad: float,
    origen: str,
    id_orden: str | None = None,
    id_sitio: int | str | None = None,
    comentario: str = ""
) -> dict:
    signo = 0
    impacto_on_hand = 0.0
    impacto_reserved = 0.0

    if tipo_movimiento == "LLEGADA":
        signo = 1
        impacto_on_hand = float(cantidad)
    elif tipo_movimiento == "RESERVA":
        signo = 0
        impacto_reserved = float(cantidad)
    elif tipo_movimiento == "CONSUMO":
        signo = -1
        impacto_on_hand = -float(cantidad)
        impacto_reserved = -float(cantidad)
    else:
        raise ValueError(f"tipo_movimiento no soportado: {tipo_movimiento}")

    return {
        "id_transaccion": str(uuid.uuid4()),
        "year_week": year_week,
        "tipo_movimiento": tipo_movimiento,
        "id_referencia": id_referencia,
        "referencia": referencia,
        "unidad": unidad,
        "cantidad": float(cantidad),
        "signo": signo,
        "impacto_on_hand": float(impacto_on_hand),
        "impacto_reserved": float(impacto_reserved),
        "id_orden": id_orden,
        "id_sitio": id_sitio,
        "origen": origen,
        "comentario": comentario,
        "fecha_proceso": random_datetime_in_year_week(
            year_week=year_week,
            seed_key=f"{tipo_movimiento}|{id_orden}|{id_referencia}|{cantidad}"
        ),
    }


def recalc_available(df_inv: pd.DataFrame) -> None:
    df_inv["available"] = (df_inv["on_hand"] - df_inv["reserved"]).astype(float)


def apply_transaction_to_inventory(df_inv: pd.DataFrame, tx: dict) -> None:
    mask = df_inv["id_referencia"] == tx["id_referencia"]
    if not mask.any():
        df_inv.loc[len(df_inv)] = {
            "id_referencia": tx["id_referencia"],
            "referencia": tx["referencia"],
            "unidad": tx["unidad"],
            "on_hand": 0.0,
            "reserved": 0.0,
            "available": 0.0,
            "year_week_corte": pd.NA,
            "fecha_proceso": tx["fecha_proceso"],
        }
        mask = df_inv["id_referencia"] == tx["id_referencia"]

    df_inv.loc[mask, "on_hand"] = df_inv.loc[mask, "on_hand"].astype(float) + tx["impacto_on_hand"]
    df_inv.loc[mask, "reserved"] = df_inv.loc[mask, "reserved"].astype(float) + tx["impacto_reserved"]

    df_inv.loc[mask, "on_hand"] = df_inv.loc[mask, "on_hand"].clip(lower=0.0)
    df_inv.loc[mask, "reserved"] = df_inv.loc[mask, "reserved"].clip(lower=0.0)

    recalc_available(df_inv)


def try_reserve_for_orders(
    year_week: str,
    df_inv: pd.DataFrame,
    df_ord: pd.DataFrame,
    df_det: pd.DataFrame,
    order_ids: list[str],
    tx_acc: list[dict],
) -> None:
    """Reserva respetando available, actualiza detalle + cabecera."""
    for id_orden in order_ids:
        det_mask = df_det["id_orden"] == id_orden
        det = df_det.loc[det_mask].copy()
        if det.empty:
            continue

        for _, row in det.iterrows():
            id_ref = row["id_referencia"]
            pendiente = float(row["cantidad_pendiente"])

            if pendiente <= 0:
                continue

            inv_mask = df_inv["id_referencia"] == id_ref
            if not inv_mask.any():
                continue

            available = float(df_inv.loc[inv_mask, "available"].iloc[0])
            asignar = min(pendiente, available)
            if asignar <= 0:
                continue

            tx = add_transaction(
                year_week=year_week,
                tipo_movimiento="RESERVA",
                id_referencia=id_ref,
                referencia=row["referencia"],
                unidad=row["unidad"],
                cantidad=asignar,
                origen="Configuracion_Real",
                id_orden=id_orden,
                id_sitio=row["id_sitio"],
                comentario="Reserva parcial o total para orden",
            )
            tx_acc.append(tx)
            apply_transaction_to_inventory(df_inv, tx)

            # Update detalle
            m = det_mask & (df_det["id_referencia"] == id_ref)
            df_det.loc[m, "cantidad_asignada"] = df_det.loc[m, "cantidad_asignada"].astype(float) + asignar
            df_det.loc[m, "cantidad_pendiente"] = df_det.loc[m, "cantidad_pendiente"].astype(float) - asignar
            df_det.loc[m, "week_ultima_reserva"] = year_week

        det_after = df_det.loc[det_mask].copy()
        df_det.loc[det_mask, "estado_item"] = det_after["cantidad_pendiente"].apply(
            lambda x: "COMPLETO" if float(x) <= 0 else "PARCIAL"
        )

        tot_solic = float(det_after["cantidad_solicitada"].sum())
        tot_asig = float(det_after["cantidad_asignada"].sum())
        tot_pend = float(det_after["cantidad_pendiente"].sum())

        ord_mask = df_ord["id_orden"] == id_orden
        df_ord.loc[ord_mask, "total_solicitado"] = tot_solic
        df_ord.loc[ord_mask, "total_asignado"] = tot_asig
        df_ord.loc[ord_mask, "total_pendiente"] = tot_pend

        if tot_asig == 0:
            estado = "PENDIENTE"
        elif tot_pend > 0:
            estado = "PARCIAL"
        else:
            estado = "COMPLETA"

        df_ord.loc[ord_mask, "estado_orden"] = estado

        if (df_ord.loc[ord_mask, "week_primera_reserva"].isna().any()) and (tot_asig > 0):
            df_ord.loc[ord_mask, "week_primera_reserva"] = year_week

        if (df_ord.loc[ord_mask, "week_completado"].isna().any()) and (estado == "COMPLETA"):
            df_ord.loc[ord_mask, "week_completado"] = year_week


def consume_completed_orders(
    year_week: str,
    df_inv: pd.DataFrame,
    df_ord: pd.DataFrame,
    df_det: pd.DataFrame,
    tx_acc: list[dict],
) -> None:
    """Consume solo órdenes COMPLETAS no consumidas."""
    mask = (df_ord["estado_orden"] == "COMPLETA") & (df_ord["week_consumido"].isna())
    ordenes_a_consumir = df_ord.loc[mask, "id_orden"].tolist()

    for id_orden in ordenes_a_consumir:
        det_mask = df_det["id_orden"] == id_orden
        det = df_det.loc[det_mask].copy()
        if det.empty:
            continue

        for _, row in det.iterrows():
            qty = float(row["cantidad_asignada"])
            if qty <= 0:
                continue

            tx = add_transaction(
                year_week=year_week,
                tipo_movimiento="CONSUMO",
                id_referencia=row["id_referencia"],
                referencia=row["referencia"],
                unidad=row["unidad"],
                cantidad=qty,
                origen="Configuracion_Real",
                id_orden=id_orden,
                id_sitio=row["id_sitio"],
                comentario="Despacho a sitio con HW completo",
            )
            tx_acc.append(tx)
            apply_transaction_to_inventory(df_inv, tx)

        ord_mask = df_ord["id_orden"] == id_orden
        df_ord.loc[ord_mask, "estado_orden"] = "CONSUMIDA"
        df_ord.loc[ord_mask, "week_consumido"] = year_week

#%% ===============================================================
# 8.1) CONTAMINACIÓN (RAW) Y LIMPIEZA (CLEAN)
# ===============================================================

GLOBAL_SEED = 20260215

def contaminate_weekly(df: pd.DataFrame, wk: str, profile: str = "light") -> pd.DataFrame:
    """
    Genera versión RAW contaminada del dataframe semanal.
    Contaminación reproducible por semana.
    """

    df_raw = df.copy()

    # ---------------------------
    # Configuración de tasas
    # ---------------------------
    if profile == "light":
        rate_date = 0.10
        rate_year_week = 0.02
        rate_text = 0.15
    else:
        rate_date = 0.20
        rate_year_week = 0.05
        rate_text = 0.25

    rnd = random.Random(f"CONTAMINATE|{wk}|{GLOBAL_SEED}")

    # ---------------------------
    # 1) fecha_proceso
    # ---------------------------
    if "fecha_proceso" in df_raw.columns:
        for i in df_raw.index:
            if rnd.random() < rate_date:
                val = str(df_raw.at[i, "fecha_proceso"])
                try:
                    dt = pd.to_datetime(val)
                    fmt_choice = rnd.choice([
                        "%Y/%m/%d %H:%M:%S",
                        "%d-%m-%Y %H:%M:%S",
                        "%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%d %H:%M"
                    ])
                    df_raw.at[i, "fecha_proceso"] = dt.strftime(fmt_choice)
                except:
                    pass

    # ---------------------------
    # 2) year_week
    # ---------------------------
    if "year_week" in df_raw.columns:
        for i in df_raw.index:
            if rnd.random() < rate_year_week:
                yw = str(df_raw.at[i, "year_week"])
                try:
                    year, week = yw.split("_")
                    variant = rnd.choice(["no_zero", "dash", "slash"])
                    if variant == "no_zero":
                        df_raw.at[i, "year_week"] = f"{year}_{int(week)}"
                    elif variant == "dash":
                        df_raw.at[i, "year_week"] = f"{year}-{week}"
                    elif variant == "slash":
                        df_raw.at[i, "year_week"] = f"{year}/{week}"
                except:
                    pass

    # ---------------------------
    # 3) Textos
    # ---------------------------
    text_cols = ["referencia", "nombre_sitio", "unidad", "origen", "comentario"]

    for col in text_cols:
        if col in df_raw.columns:
            for i in df_raw.index:
                if rnd.random() < rate_text:
                    val = str(df_raw.at[i, col])
                    variant = rnd.choice(["spaces", "upper", "lower", "double_space"])
                    if variant == "spaces":
                        df_raw.at[i, col] = f" {val} "
                    elif variant == "upper":
                        df_raw.at[i, col] = val.upper()
                    elif variant == "lower":
                        df_raw.at[i, col] = val.lower()
                    elif variant == "double_space":
                        df_raw.at[i, col] = val.replace(" ", "  ")

    return df_raw


def standardize_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y estandariza versión RAW → CLEAN
    """

    df_clean = df.copy()

    # ---------------------------
    # 1) Normalizar fecha_proceso
    # ---------------------------
    if "fecha_proceso" in df_clean.columns:
        df_clean["fecha_proceso"] = pd.to_datetime(
            df_clean["fecha_proceso"],
            errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")

    # ---------------------------
    # 2) Normalizar year_week
    # ---------------------------
    if "year_week" in df_clean.columns:
        def fix_yw(yw):
            try:
                yw = str(yw).replace("-", "_").replace("/", "_")
                year, week = yw.split("_")
                return f"{int(year)}_{int(week):02d}"
            except:
                return None

        df_clean["year_week"] = df_clean["year_week"].apply(fix_yw)

    # ---------------------------
    # 3) Textos
    # ---------------------------
    text_cols = ["referencia", "nombre_sitio", "unidad", "origen", "comentario"]

    for col in text_cols:
        if col in df_clean.columns:
            df_clean[col] = (
                df_clean[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

    # Reglas específicas
    if "referencia" in df_clean.columns:
        df_clean["referencia"] = df_clean["referencia"].str.upper()

    if "unidad" in df_clean.columns:
        df_clean["unidad"] = df_clean["unidad"].str.lower()

    return df_clean


#%% ===============================================================
# 9) EJECUCIÓN PRINCIPAL SEMANA A SEMANA + CSV + CHECKPOINT
# ===============================================================

# Index para acelerar
llegadas_by_week = {
    wk: df_llegadas.loc[df_llegadas["year_week"] == wk].copy()
    for wk in horizon_weeks
}

ordenes_by_week = {
    wk: df_ordenes.loc[df_ordenes["year_week_necesidad"] == wk, "id_orden"].tolist()
    for wk in horizon_weeks
}

inventario_semanal_acc = []        # snapshots semanales (para export opcional)
transacciones_hist_acc = []        # histórico completo (para export opcional)

for wk in horizon_weeks:
    # Acumulador semanal (esto es lo que se exporta a CSV)
    tx_wk_acc: list[dict] = []

    # Snapshot inicio semana
    inv_inicio = df_inventario_ref[["id_referencia", "on_hand", "reserved", "available"]].copy()
    inv_inicio = inv_inicio.rename(columns={
        "on_hand": "on_hand_inicio_semana",
        "reserved": "reserved_inicio_semana",
        "available": "available_inicio_semana",
    })

    # 1) LLEGADAS
    df_wk_llegadas = llegadas_by_week.get(wk, pd.DataFrame())
    if not df_wk_llegadas.empty:
        for _, row in df_wk_llegadas.iterrows():
            tx = add_transaction(
                year_week=wk,
                tipo_movimiento="LLEGADA",
                id_referencia=row["id_referencia"],
                referencia=row["referencia"],
                unidad=row["unidad"],
                cantidad=float(row["cantidad_llegada"]),
                origen="Pedidos",
                comentario="Ingreso de hardware por pedido",
            )
            tx_wk_acc.append(tx)
            apply_transaction_to_inventory(df_inventario_ref, tx)

    # 2) Backlog (órdenes anteriores pendientes/parciales)
    backlog_mask = (df_ordenes["estado_orden"].isin(["PENDIENTE", "PARCIAL"])) & (
        df_ordenes["year_week_necesidad"].apply(year_week_to_key) < year_week_to_key(wk)
    )
    backlog_ids = df_ordenes.loc[backlog_mask, "id_orden"].tolist()

    if backlog_ids:
        try_reserve_for_orders(
            year_week=wk,
            df_inv=df_inventario_ref,
            df_ord=df_ordenes,
            df_det=df_ordenes_detalle,
            order_ids=backlog_ids,
            tx_acc=tx_wk_acc,
        )

    # 3) Órdenes nuevas de la semana
    new_order_ids = ordenes_by_week.get(wk, [])
    if new_order_ids:
        try_reserve_for_orders(
            year_week=wk,
            df_inv=df_inventario_ref,
            df_ord=df_ordenes,
            df_det=df_ordenes_detalle,
            order_ids=new_order_ids,
            tx_acc=tx_wk_acc,
        )

    # 4) CONSUMO (solo completas)
    consume_completed_orders(
        year_week=wk,
        df_inv=df_inventario_ref,
        df_ord=df_ordenes,
        df_det=df_ordenes_detalle,
        tx_acc=tx_wk_acc,
    )

    # Snapshot fin de semana
    inv_fin = df_inventario_ref[["id_referencia", "on_hand", "reserved", "available"]].copy()
    inv_fin = inv_fin.rename(columns={
        "on_hand": "on_hand_fin_semana",
        "reserved": "reserved_fin_semana",
        "available": "available_fin_semana",
    })

    # Movimientos semana
    df_tx_wk = pd.DataFrame(tx_wk_acc)

    if df_tx_wk.empty:
        movs = pd.DataFrame(columns=["id_referencia", "llegadas_qty_semana", "reservas_qty_semana", "consumos_qty_semana"])
    else:
        qtys = (
            df_tx_wk
            .groupby(["id_referencia", "tipo_movimiento"], as_index=False)
            .agg(cantidad=("cantidad", "sum"))
        )
        qtys = (
            qtys
            .pivot(index="id_referencia", columns="tipo_movimiento", values="cantidad")
            .fillna(0.0)
            .reset_index()
        )
        movs = qtys.rename(columns={
            "LLEGADA": "llegadas_qty_semana",
            "RESERVA": "reservas_qty_semana",
            "CONSUMO": "consumos_qty_semana",
        })

    # Snapshot semanal (para export opcional y/o tabla opcional)
    snap = (
        df_inventario_ref[["id_referencia", "referencia", "unidad"]]
        .merge(inv_inicio, on="id_referencia", how="left")
        .merge(inv_fin, on="id_referencia", how="left")
        .merge(movs, on="id_referencia", how="left")
    )
    snap["year_week"] = wk
    snap["fecha_proceso"] = random_datetime_in_year_week(year_week=wk, seed_key=f"SNAP|{wk}")
    inventario_semanal_acc.append(snap)

    # Corte procesado
    df_inventario_ref["year_week_corte"] = wk

    # ------------------------------------------------------------
    # CIERRE DE SEMANA (1): CSV semanal -> Data Lake (overwrite)
    # ------------------------------------------------------------
    tx_cols = [
        "id_transaccion", "year_week", "tipo_movimiento",
        "id_referencia", "referencia", "unidad",
        "cantidad", "id_orden", "id_sitio", "origen", "comentario", "fecha_proceso"
    ]

    if df_tx_wk.empty:
        df_tx_wk_export = pd.DataFrame(columns=tx_cols)
    else:
        for c in tx_cols:
            if c not in df_tx_wk.columns:
                df_tx_wk[c] = None
        df_tx_wk_export = df_tx_wk[tx_cols].copy()

    year_str, week_str = wk.split("_")
    year = int(year_str)
    week = int(week_str)

    blob_path = (
        f"{DATALAKE_BASE_PATH}/"
        f"year={year}/week={week:02d}/"
        f"transacciones_{year}_{week:02d}.csv"
    )

    # ------------------------------------------------------------
    # CONTAMINACIÓN → RAW
    # ------------------------------------------------------------
    df_tx_wk_raw = contaminate_weekly(df_tx_wk_export, wk, profile="light")

    year_str, week_str = wk.split("_")
    year = int(year_str)
    week = int(week_str)

    raw_path = (
        f"{DATALAKE_BASE_PATH}/raw/"
        f"year={year}/week={week:02d}/"
        f"transacciones_{year}_{week:02d}.csv"
    )

    upload_df_as_csv(df_tx_wk_raw, raw_path)
    print(f"[OK] RAW semana {wk} subido: {raw_path}")

    # ------------------------------------------------------------
    # LIMPIEZA → CLEAN
    # ------------------------------------------------------------
    df_tx_wk_clean = standardize_weekly(df_tx_wk_raw)

    clean_path = (
        f"{DATALAKE_BASE_PATH}/clean/"
        f"year={year}/week={week:02d}/"
        f"transacciones_{year}_{week:02d}.csv"
    )

    upload_df_as_csv(df_tx_wk_clean, clean_path)
    print(f"[OK] CLEAN semana {wk} subido: {clean_path}")


    # ------------------------------------------------------------
    # CIERRE DE SEMANA (2): Persistir estado + checkpoint en Azure SQL
    # ------------------------------------------------------------
    persist_state(engine, df_inventario_ref, df_ordenes, df_ordenes_detalle)
    set_checkpoint(engine, last_year_week_processed=wk, status="OK")
    print(f"[OK] Checkpoint actualizado: {wk}")

    # ------------------------------------------------------------
    # CIERRE DE SEMANA (3): opcional facts (append)
    # ------------------------------------------------------------
    append_optional_facts(engine, df_tx_wk_export, snap)

    # Acumuladores para export opcional
    if not df_tx_wk.empty:
        transacciones_hist_acc.extend(tx_wk_acc)


#%% ===============================================================
# 10) EXPORT OPCIONAL (LOCAL) – solo si EXPORT_EXCEL=True
# ===============================================================

if EXPORT_EXCEL:
    df_transacciones = pd.DataFrame(transacciones_hist_acc)
    df_inventario_semanal = pd.concat(inventario_semanal_acc, ignore_index=True)

    cols_tx = [
        "id_transaccion", "year_week", "tipo_movimiento",
        "id_referencia", "referencia", "unidad",
        "cantidad", "signo", "impacto_on_hand", "impacto_reserved",
        "id_orden", "id_sitio", "origen", "comentario", "fecha_proceso"
    ]
    cols_tx = [c for c in cols_tx if c in df_transacciones.columns]
    df_transacciones = df_transacciones[cols_tx]

    cols_inv_wk = [
        "year_week", "id_referencia", "referencia", "unidad",
        "llegadas_qty_semana", "reservas_qty_semana", "consumos_qty_semana",
        "on_hand_inicio_semana", "reserved_inicio_semana", "available_inicio_semana",
        "on_hand_fin_semana", "reserved_fin_semana", "available_fin_semana",
        "fecha_proceso"
    ]
    cols_inv_wk = [c for c in cols_inv_wk if c in df_inventario_semanal.columns]
    df_inventario_semanal = df_inventario_semanal[cols_inv_wk]

    df_ordenes_resumen = df_ordenes.copy()
    df_ordenes_detalle_out = df_ordenes_detalle.copy()

    df_ordenes_resumen["fecha_proceso"] = df_ordenes_resumen["year_week_necesidad"].apply(
        lambda yw: random_datetime_in_year_week(str(yw), seed_key=f"ORDEN|{yw}")
    )

    df_transacciones.to_excel(RUTA_TRANSACCIONES_OUT, index=False, sheet_name="transacciones")
    df_inventario_semanal.to_excel(RUTA_INVENTARIO_SEMANAL_OUT, index=False, sheet_name="inventario_semanal")
    df_ordenes_resumen.to_excel(RUTA_ORDENES_RESUMEN_OUT, index=False, sheet_name="ordenes_resumen")
    df_ordenes_detalle_out.to_excel(RUTA_ORDENES_DETALLE_OUT, index=False, sheet_name="ordenes_detalle")

    print(f"[OK] Export Excel: {RUTA_TRANSACCIONES_OUT}")
    print(f"[OK] Export Excel: {RUTA_INVENTARIO_SEMANAL_OUT}")
    print(f"[OK] Export Excel: {RUTA_ORDENES_RESUMEN_OUT}")
    print(f"[OK] Export Excel: {RUTA_ORDENES_DETALLE_OUT}")

print("[DONE] Simulación completada.")
