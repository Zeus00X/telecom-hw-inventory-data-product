#%%
# step_02_simulate_inventory_dynamics.py
# Simulación robusta de inventario semana a semana:
# 1) LLEGADAS: incrementan on_hand
# 2) RESERVAS: incrementan reserved, no afectan on_hand
# 3) CONSUMO: se ejecuta solo cuando la orden queda 100% completa; descuenta on_hand y reserved
#
# Entradas (Fase 1):
# - 01_simulation/output/01_sitios_x_configuracion_real.xlsx
# - 01_simulation/output/01_pedidos_desde_proyeccion.xlsx
#
# Salidas (Fase 2):
# - 01_simulation/output/02_transacciones_inventario.xlsx
# - 01_simulation/output/02_inventario_semanal.xlsx
# - 01_simulation/output/02_ordenes_resumen.xlsx
# - 01_simulation/output/02_ordenes_detalle.xlsx

import os
import uuid
import random
from datetime import datetime, time

import pandas as pd


#%%
# Parámetros generales del script

# Entradas – Fase 1
RUTA_SITIOS_REAL = "01_simulation/output/01_sitios_x_configuracion_real.xlsx"
RUTA_PEDIDOS = "01_simulation/output/01_pedidos_desde_proyeccion.xlsx"

CARPETA_SALIDA = "01_simulation/output"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# Salidas – Fase 2
RUTA_TRANSACCIONES_OUT = f"{CARPETA_SALIDA}/02_transacciones_inventario.xlsx"
RUTA_INVENTARIO_SEMANAL_OUT = f"{CARPETA_SALIDA}/02_inventario_semanal.xlsx"
RUTA_ORDENES_RESUMEN_OUT = f"{CARPETA_SALIDA}/02_ordenes_resumen.xlsx"
RUTA_ORDENES_DETALLE_OUT = f"{CARPETA_SALIDA}/02_ordenes_detalle.xlsx"

#%% Seleccionar una fecha aleatorio entre el year_week

def random_datetime_in_year_week(year_week: str, seed_key: str | None = None) -> str:
    """Retorna una fecha aleatoria dentro de la semana ISO indicada por 'YYYY_WW'.

    Regla:
    - Se toma el lunes ISO de la semana y se suma un offset aleatorio 0..6 días.
    - La hora se genera aleatoria (0..23), minuto (0..59), segundo (0..59).
    - Se retorna en formato '%Y-%m-%d %H:%M:%S'.

    Nota:
    - seed_key permite reproducibilidad. Si se pasa, la aleatoriedad queda fija
      para el mismo (year_week + seed_key).
    """
    if not isinstance(year_week, str) or "_" not in year_week:
        raise ValueError("year_week debe tener formato 'YYYY_WW' (por ejemplo '2025_01').")

    y_str, w_str = year_week.split("_")
    year = int(y_str)
    week = int(w_str)

    # Lunes ISO de la semana
    monday = pd.to_datetime(f"{year}-W{week:02d}-1", format="%G-W%V-%u").to_pydatetime()

    # Semilla opcional para reproducibilidad
    if seed_key is not None:
        rnd = random.Random(f"{year_week}|{seed_key}")
    else:
        rnd = random

    day_offset = rnd.randint(0, 6)
    hour = rnd.randint(0, 23)
    minute = rnd.randint(0, 59)
    second = rnd.randint(0, 59)

    dt = monday + pd.Timedelta(days=day_offset)
    dt = dt.replace(hour=hour, minute=minute, second=second, microsecond=0)

    return dt.strftime("%Y-%m-%d %H:%M:%S")




#%%
# Utilidades de tiempo: year_week como clave ordenable
# Se asume formato 'YYYY_WW' donde WW tiene cero a la izquierda

def year_week_to_key(year_week: str) -> int:
    """Convierte 'YYYY_WW' a entero YYYY*100+WW para ordenar."""
    year, week = year_week.split("_")
    return int(year) * 100 + int(week)

def key_to_year_week(key: int) -> str:
    """Convierte entero YYYY*100+WW a 'YYYY_WW'."""
    year = key // 100
    week = key % 100
    return f"{year}_{week:02d}"


#%%
# Horizonte continuo: lista completa desde min_week_sim hasta max_week_sim
# Regla: se incrementa semana a semana, manejando el rollover de semana 52/53 según corresponda

def _iso_weeks_in_year(year: int) -> int:
    """Retorna el número de semanas ISO del año (52 o 53)."""
    # ISO week: la semana 53 existe si el 28 de diciembre cae en semana 53
    return pd.Timestamp(year=year, month=12, day=28).isocalendar().week


def build_horizon_year_weeks(min_week_sim: str, max_week_sim: str) -> list[str]:
    """Construye un horizonte continuo de semanas ISO entre min_week_sim y max_week_sim (incluyentes).

    Entradas:
    - min_week_sim: 'YYYY_WW'
    - max_week_sim: 'YYYY_WW'

    Salida:
    - Lista ordenada y continua de year_week entre ambos extremos.
    """
    if not isinstance(min_week_sim, str) or "_" not in min_week_sim:
        raise ValueError("min_week_sim debe tener formato 'YYYY_WW' (por ejemplo '2025_01').")

    if not isinstance(max_week_sim, str) or "_" not in max_week_sim:
        raise ValueError("max_week_sim debe tener formato 'YYYY_WW' (por ejemplo '2025_52').")

    y0, w0 = min_week_sim.split("_")
    y1, w1 = max_week_sim.split("_")

    year = int(y0)
    week = int(w0)

    end_year = int(y1)
    end_week = int(w1)

    # Validación básica de rango
    if (year > end_year) or (year == end_year and week > end_week):
        raise ValueError("min_week_sim debe ser menor o igual que max_week_sim.")

    horizon = []
    while True:
        horizon.append(f"{year}_{week:02d}")

        if year == end_year and week == end_week:
            break

        # Avance de semana con rollover ISO
        weeks_in_year = _iso_weeks_in_year(year)
        week += 1
        if week > weeks_in_year:
            year += 1
            week = 1

    return horizon


#%%
# Lectura de insumos

from utils.azure_sql_io import build_engine_from_env, read_table

engine = build_engine_from_env()

df_sitios_real = read_table(engine, "stg", "sitios_x_configuracion_real")
df_pedidos = read_table(engine, "stg", "pedidos_desde_proyeccion")


#df_sitios_real = pd.read_excel(RUTA_SITIOS_REAL)
#df_pedidos = pd.read_excel(RUTA_PEDIDOS)

# Normalización mínima de tipos
df_sitios_real["year_week"] = df_sitios_real["year_week"].astype(str)
df_pedidos["year_week_pedido"] = df_pedidos["year_week_pedido"].astype(str)
df_pedidos["year_week_disponible"] = df_pedidos["year_week_disponible"].astype(str)


print(df_sitios_real.shape)
print(df_pedidos.shape)
print(df_sitios_real.head(2))
print(df_pedidos.head(2))



#%%
# Preparación de llegadas por semana y referencia (input para transacciones tipo LLEGADA)
# Grano: year_week_disponible + id_referencia

df_llegadas = (
    df_pedidos
    .groupby(["year_week_disponible", "id_referencia", "referencia", "unidad"], as_index=False)
    .agg(cantidad_llegada=("cantidad_pedido", "sum"))
)

# Renombre para estandarizar con year_week de simulación
df_llegadas = df_llegadas.rename(columns={"year_week_disponible": "year_week"})


#%%
# Preparación de demanda: creación de órdenes por sitio y semana, desde consumo real (cantidad_real)
# Regla: la necesidad del sitio para una referencia es cantidad_real (simulada) y se solicita en la semana year_week.

# 1) Detalle de orden: grano id_sitio + year_week + id_referencia
df_ordenes_detalle = (
    df_sitios_real
    .groupby(
        ["id_sitio", "nombre_sitio", "region", "year_week", "id_referencia", "referencia", "unidad"],
        as_index=False
    )
    .agg(cantidad_solicitada=("cantidad_real", "sum"))
)

# 2) Identificador de orden
df_ordenes_detalle["id_orden"] = (
    df_ordenes_detalle["id_sitio"].astype(str) + "_" + df_ordenes_detalle["year_week"].astype(str)
)

# 3) Campos operativos del detalle
df_ordenes_detalle["cantidad_asignada"] = 0.0
df_ordenes_detalle["cantidad_pendiente"] = df_ordenes_detalle["cantidad_solicitada"].astype(float)
df_ordenes_detalle["estado_item"] = "PENDIENTE"
df_ordenes_detalle["week_ultima_reserva"] = pd.NA

# 4) Cabecera de orden
df_ordenes = (
    df_ordenes_detalle
    .groupby(["id_orden", "id_sitio", "nombre_sitio", "region", "year_week"], as_index=False)
    .agg(
        total_referencias=("id_referencia", "nunique"),
        total_solicitado=("cantidad_solicitada", "sum"),
        total_asignado=("cantidad_asignada", "sum"),
        total_pendiente=("cantidad_pendiente", "sum"),
    )
)

df_ordenes = df_ordenes.rename(columns={"year_week": "year_week_necesidad"})
df_ordenes["estado_orden"] = "PENDIENTE"
df_ordenes["week_primera_reserva"] = pd.NA
df_ordenes["week_completado"] = pd.NA
df_ordenes["week_consumido"] = pd.NA
df_ordenes["fecha_proceso"] = pd.NA

#%%
# Construcción del horizonte de simulación

MIN_WEEK_SIM = "2025_01"
MAX_WEEK_SIM = "2025_52"

horizon_weeks = build_horizon_year_weeks(
    min_week_sim=MIN_WEEK_SIM,
    max_week_sim=MAX_WEEK_SIM
)


#%%
# Inicialización del inventario por referencia
# Regla: on_hand inicia en 0; las llegadas de 2025_1 ya incluyen el arranque.

refs_from_llegadas = df_llegadas[["id_referencia", "referencia", "unidad"]].drop_duplicates()
refs_from_demanda = df_ordenes_detalle[["id_referencia", "referencia", "unidad"]].drop_duplicates()

df_refs = (
    pd.concat([refs_from_llegadas, refs_from_demanda], ignore_index=True)
    .drop_duplicates(subset=["id_referencia"])
)

df_inventario_ref = df_refs.copy()
df_inventario_ref["on_hand"] = 0.0
df_inventario_ref["reserved"] = 0.0
df_inventario_ref["available"] = 0.0
df_inventario_ref["year_week_corte"] = pd.NA
df_inventario_ref["fecha_proceso"] = pd.NA


#%%
# Estructuras vacías de salida
# df_transacciones: histórico de eventos
# df_inventario_semanal: snapshot por semana y referencia

transacciones = []
inventario_semanal = []


#%%
# Funciones de lógica de negocio: creación de transacciones y aplicación sobre inventario

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
    """Crea un registro estándar de transacción."""
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
        "fecha_proceso":  random_datetime_in_year_week(
        year_week=year_week,
        seed_key=f"{tipo_movimiento}|{id_orden}|{id_referencia}|{cantidad}"
    ),
    }


def recalc_available(df_inv: pd.DataFrame) -> None:
    """Recalcula available en el inventario."""
    df_inv["available"] = (df_inv["on_hand"] - df_inv["reserved"]).astype(float)


def apply_transaction_to_inventory(df_inv: pd.DataFrame, tx: dict) -> None:
    """Aplica una transacción al inventario por referencia."""
    mask = df_inv["id_referencia"] == tx["id_referencia"]
    if not mask.any():
        # Creación de referencia si no existe
        df_inv.loc[len(df_inv)] = {
            "id_referencia": tx["id_referencia"],
            "referencia": tx["referencia"],
            "unidad": tx["unidad"],
            "on_hand": 0.0,
            "reserved": 0.0,
            "available": 0.0,
            "year_week_corte": pd.NA,
            "fecha_proceso":  tx["fecha_proceso"],
        }
        mask = df_inv["id_referencia"] == tx["id_referencia"]

    df_inv.loc[mask, "on_hand"] = df_inv.loc[mask, "on_hand"].astype(float) + tx["impacto_on_hand"]
    df_inv.loc[mask, "reserved"] = df_inv.loc[mask, "reserved"].astype(float) + tx["impacto_reserved"]

    # Reglas de integridad básicas
    df_inv.loc[mask, "on_hand"] = df_inv.loc[mask, "on_hand"].clip(lower=0.0)
    df_inv.loc[mask, "reserved"] = df_inv.loc[mask, "reserved"].clip(lower=0.0)

    recalc_available(df_inv)


#%%
# Funciones de lógica: asignación (RESERVA) y consumo (CONSUMO) con regla de completitud

def try_reserve_for_orders(
    year_week: str,
    df_inv: pd.DataFrame,
    df_ord: pd.DataFrame,
    df_det: pd.DataFrame,
    order_ids: list[str],
) -> None:
    """Intenta reservar inventario para un conjunto de órdenes, respetando available.
    Esta función:
    - Genera transacciones RESERVA
    - Actualiza df_inventario_ref (reserved)
    - Actualiza df_ordenes_detalle (asignada, pendiente, estado_item, week_ultima_reserva)
    - Actualiza df_ordenes (totales, estado_orden, week_primera_reserva, week_completado)
    """
    global transacciones

    for id_orden in order_ids:
        det_mask = df_det["id_orden"] == id_orden
        det = df_det.loc[det_mask].copy()

        if det.empty:
            continue

        # Reservar por referencia
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

            # Transacción de reserva
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
            transacciones.append(tx)
            apply_transaction_to_inventory(df_inv, tx)

            # Actualización del detalle
            df_det.loc[det_mask & (df_det["id_referencia"] == id_ref), "cantidad_asignada"] = (
                df_det.loc[det_mask & (df_det["id_referencia"] == id_ref), "cantidad_asignada"].astype(float) + asignar
            )
            df_det.loc[det_mask & (df_det["id_referencia"] == id_ref), "cantidad_pendiente"] = (
                df_det.loc[det_mask & (df_det["id_referencia"] == id_ref), "cantidad_pendiente"].astype(float) - asignar
            )
            df_det.loc[det_mask & (df_det["id_referencia"] == id_ref), "week_ultima_reserva"] = year_week

        # Recalcular estado por ítem
        det_after = df_det.loc[det_mask].copy()
        df_det.loc[det_mask, "estado_item"] = det_after["cantidad_pendiente"].apply(
            lambda x: "COMPLETO" if float(x) <= 0 else "PARCIAL"
        )

        # Recalcular cabecera de orden
        tot_solic = float(det_after["cantidad_solicitada"].sum())
        tot_asig = float(det_after["cantidad_asignada"].sum())
        tot_pend = float(det_after["cantidad_pendiente"].sum())

        ord_mask = df_ord["id_orden"] == id_orden
        df_ord.loc[ord_mask, "total_solicitado"] = tot_solic
        df_ord.loc[ord_mask, "total_asignado"] = tot_asig
        df_ord.loc[ord_mask, "total_pendiente"] = tot_pend

        # Estado de la orden
        if tot_asig == 0:
            estado = "PENDIENTE"
        elif tot_pend > 0:
            estado = "PARCIAL"
        else:
            estado = "COMPLETA"

        df_ord.loc[ord_mask, "estado_orden"] = estado

        # Marcas de semanas clave
        if (df_ord.loc[ord_mask, "week_primera_reserva"].isna().any()) and (tot_asig > 0):
            df_ord.loc[ord_mask, "week_primera_reserva"] = year_week

        if (df_ord.loc[ord_mask, "week_completado"].isna().any()) and (estado == "COMPLETA"):
            df_ord.loc[ord_mask, "week_completado"] = year_week


def consume_completed_orders(
    year_week: str,
    df_inv: pd.DataFrame,
    df_ord: pd.DataFrame,
    df_det: pd.DataFrame,
) -> None:
    """Ejecuta CONSUMO únicamente para órdenes 100% completas y no consumidas.
    - Genera transacciones CONSUMO por referencia del detalle (cantidad_asignada)
    - Descuenta on_hand y reserved
    - Marca la orden como CONSUMIDA
    """
    global transacciones

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
            transacciones.append(tx)
            apply_transaction_to_inventory(df_inv, tx)

        ord_mask = df_ord["id_orden"] == id_orden
        df_ord.loc[ord_mask, "estado_orden"] = "CONSUMIDA"
        df_ord.loc[ord_mask, "week_consumido"] = year_week


#%%
# Ejecución principal: iteración semana a semana
# Orden:
# 1) Aplicar LLEGADAS
# 2) Completar backlog (órdenes anteriores pendientes)
# 3) Reservar para órdenes nuevas de la semana
# 4) Consumir órdenes completas (solo cuando están 100% completas)

llegadas_by_week = {
    wk: df_llegadas.loc[df_llegadas["year_week"] == wk].copy()
    for wk in horizon_weeks
}

ordenes_by_week = {
    wk: df_ordenes.loc[df_ordenes["year_week_necesidad"] == wk, "id_orden"].tolist()
    for wk in horizon_weeks
}

for wk in horizon_weeks:
    # Snapshot de inicio de semana
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
            transacciones.append(tx)
            apply_transaction_to_inventory(df_inventario_ref, tx)

    # 2) Backlog: órdenes de semanas anteriores con pendiente > 0
    backlog_mask = (df_ordenes["estado_orden"].isin(["PENDIENTE", "PARCIAL"])) & (
        df_ordenes["year_week_necesidad"].apply(year_week_to_key) < year_week_to_key(wk)
    )
    backlog_ids = df_ordenes.loc[backlog_mask, "id_orden"].tolist()

    if len(backlog_ids) > 0:
        try_reserve_for_orders(
            year_week=wk,
            df_inv=df_inventario_ref,
            df_ord=df_ordenes,
            df_det=df_ordenes_detalle,
            order_ids=backlog_ids,
        )

    # 3) Órdenes nuevas de la semana
    new_order_ids = ordenes_by_week.get(wk, [])
    if len(new_order_ids) > 0:
        try_reserve_for_orders(
            year_week=wk,
            df_inv=df_inventario_ref,
            df_ord=df_ordenes,
            df_det=df_ordenes_detalle,
            order_ids=new_order_ids,
        )

    # 4) CONSUMO solo para órdenes completas
    consume_completed_orders(
        year_week=wk,
        df_inv=df_inventario_ref,
        df_ord=df_ordenes,
        df_det=df_ordenes_detalle,
    )

    # Snapshot fin de semana y agregados de movimientos semanales
    inv_fin = df_inventario_ref[["id_referencia", "on_hand", "reserved", "available"]].copy()
    inv_fin = inv_fin.rename(columns={
        "on_hand": "on_hand_fin_semana",
        "reserved": "reserved_fin_semana",
        "available": "available_fin_semana",
    })

    # Sumar movimientos de la semana desde transacciones
    df_tx_wk = pd.DataFrame([t for t in transacciones if t.get("year_week") == wk])

    if df_tx_wk.empty:
        movs = pd.DataFrame(columns=[
            "id_referencia", "llegadas_qty_semana", "reservas_qty_semana", "consumos_qty_semana"
        ])
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

    # Construcción del snapshot semanal
    snap = (
        df_inventario_ref[["id_referencia", "referencia", "unidad"]]
        .merge(inv_inicio, on="id_referencia", how="left")
        .merge(inv_fin, on="id_referencia", how="left")
        .merge(movs, on="id_referencia", how="left")
    )
    snap["year_week"] = wk
    snap["fecha_proceso"] = random_datetime_in_year_week(
        year_week=wk,
        seed_key=f"SNAP|{wk}"
    )

    inventario_semanal.append(snap)

    # Corte procesado
    df_inventario_ref["year_week_corte"] = wk

    # ------------------------------------------------------------
    # CIERRE DE SEMANA: Generar CSV de transacciones y subir a Data Lake (overwrite)
    # ------------------------------------------------------------
    # Importante: asume que ya tienes:
    #   from utils.datalake_io import upload_df_as_csv
    # y que transacciones es una lista de dicts.
    from utils.azure_datalake_io import upload_df_as_csv

    tx_cols = [
        "year_week", "tipo_movimiento",
        "id_referencia", "referencia", "unidad",
        "cantidad", "origen", "comentario"
    ]

    if df_tx_wk.empty:
        df_tx_wk_export = pd.DataFrame(columns=tx_cols)
    else:
        df_tx_wk_export = df_tx_wk.copy()
        for c in tx_cols:
            if c not in df_tx_wk_export.columns:
                df_tx_wk_export[c] = None
        df_tx_wk_export = df_tx_wk_export[tx_cols]

    year_str, week_str = wk.split("_")
    year = int(year_str)
    week = int(week_str)

    blob_path = (
        f"telecom-hw/inventory_simulation/transacciones/"
        f"year={year}/week={week:02d}/"
        f"transacciones_{year}_{week:02d}.csv"
    )

    upload_df_as_csv(df_tx_wk_export, blob_path)
    print(f"[OK] CSV semana {wk} subido (overwrite): {blob_path}")



#%%
# Ensamble final de outputs

df_transacciones = pd.DataFrame(transacciones)

df_inventario_semanal = pd.concat(inventario_semanal, ignore_index=True)

# Limpieza y orden de columnas en transacciones
cols_tx = [
    "id_transaccion", "year_week", "tipo_movimiento",
    "id_referencia", "referencia", "unidad",
    "cantidad", "signo", "impacto_on_hand", "impacto_reserved",
    "id_orden", "id_sitio", "origen", "comentario", "fecha_proceso"
]
df_transacciones = df_transacciones[cols_tx]

# Orden de columnas en inventario semanal
cols_inv_wk = [
    "year_week", "id_referencia", "referencia", "unidad",
    "llegadas_qty_semana", "reservas_qty_semana", "consumos_qty_semana",
    "on_hand_inicio_semana", "reserved_inicio_semana", "available_inicio_semana",
    "on_hand_fin_semana", "reserved_fin_semana", "available_fin_semana",
    "fecha_proceso"
]
cols_inv_wk_exist = [c for c in cols_inv_wk if c in df_inventario_semanal.columns]
df_inventario_semanal = df_inventario_semanal[cols_inv_wk_exist]

# Ordenes: outputs directos
df_ordenes_resumen = df_ordenes.copy()
df_ordenes_detalle_out = df_ordenes_detalle.copy()
df_ordenes_resumen["fecha_proceso"] = df_ordenes_resumen["year_week_necesidad"].apply(
    lambda yw: random_datetime_in_year_week(
        year_week=str(yw),
        seed_key=f"ORDEN|{yw}"
    )
)


#%%
# Exportación a Excel

df_transacciones.to_excel(RUTA_TRANSACCIONES_OUT, index=False, sheet_name="transacciones")
df_inventario_semanal.to_excel(RUTA_INVENTARIO_SEMANAL_OUT, index=False, sheet_name="inventario_semanal")
df_ordenes_resumen.to_excel(RUTA_ORDENES_RESUMEN_OUT, index=False, sheet_name="ordenes_resumen")
df_ordenes_detalle_out.to_excel(RUTA_ORDENES_DETALLE_OUT, index=False, sheet_name="ordenes_detalle")

print(f"Archivo exportado en {RUTA_TRANSACCIONES_OUT}")
print(f"Archivo exportado en {RUTA_INVENTARIO_SEMANAL_OUT}")
print(f"Archivo exportado en {RUTA_ORDENES_RESUMEN_OUT}")
print(f"Archivo exportado en {RUTA_ORDENES_DETALLE_OUT}")
