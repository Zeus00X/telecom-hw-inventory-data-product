#%%
# Imports

import numpy as np
import pandas as pd
import os


#%%
# Carga de archivos en excel

def cargar_configuracion_hw(ruta_archivo: str) -> pd.DataFrame:
    """
    Carga el archivo de configuraciones de hardware.
    """
    df_config = pd.read_excel(ruta_archivo)
    return df_config


def cargar_sitios_proyecto(ruta_archivo: str) -> pd.DataFrame:
    """
    Carga el archivo de sitios del proyecto.
    """
    df_sitios = pd.read_excel(ruta_archivo)
    return df_sitios


#%%
# Transformaciones replicables para la simulacion

def asignar_id_configuracion(df_sitios: pd.DataFrame) -> pd.DataFrame:
    """
    Diligencia id_configuracion con valores 1, 2 o 3, de forma aleatoria y replicable.
    """
    df = df_sitios.copy()
    df["id_configuracion"] = np.random.choice([1, 2, 3], size=len(df))
    return df


def asignar_altura_antena(
    df_sitios: pd.DataFrame,
    min_m: int = 15,
    max_m: int = 60,
    step_m: int = 5
) -> pd.DataFrame:
    """
    Diligencia altura_antena en metros, usando valores discretos cada step_m:
    15, 20, 25, hasta 60.
    """
    df = df_sitios.copy()
    valores_posibles = np.arange(min_m, max_m + 1, step_m)
    df["altura_antena"] = np.random.choice(valores_posibles, size=len(df))
    return df


def asignar_fc_instalacion_mensual(
    df_sitios: pd.DataFrame,
    fecha_inicio: str = "2025-01-01",
    min_mes: int = 90,
    max_mes: int = 110
) -> pd.DataFrame:
    """
    Diligencia fc_instalacion desde enero de 2025, asignando por mes entre 90 y 110 sitios.
    El ultimo mes queda con el remanente, incluso si es menor a 90.
    """
    df = df_sitios.copy()
    df["fc_instalacion"] = pd.NaT

    mes_actual = pd.Timestamp(fecha_inicio)

    while df["fc_instalacion"].isna().any():

        idx_disp = df.index[df["fc_instalacion"].isna()].to_numpy()
        remanente = len(idx_disp)

        k = np.random.randint(min_mes, max_mes + 1)
        k = min(k, remanente)

        idx_sel = np.random.choice(idx_disp, size=k, replace=False)

        days_in_month = mes_actual.days_in_month
        rand_days = np.random.randint(1, days_in_month + 1, size=k)
        fechas_mes = mes_actual + pd.to_timedelta(rand_days - 1, unit="D")

        df.loc[idx_sel, "fc_instalacion"] = fechas_mes

        mes_actual = mes_actual + pd.offsets.MonthBegin(1)

    return df

#%%
# Construccion del detalle sitio por referencia segun configuracion

def construir_sitios_x_configuracion(
    df_sitios: pd.DataFrame,
    df_config_hw: pd.DataFrame
) -> pd.DataFrame:
    """
    Construye el detalle de hardware por sitio, replicando las referencias segun id_configuracion.
    """
    df_cfg = df_config_hw.copy()

    if "unidad" not in df_cfg.columns:
        df_cfg["unidad"] = "und"
        df_cfg.loc[df_cfg["referencia"].str.contains("_M$", na=False), "unidad"] = "m"

    df = df_sitios.merge(
        df_cfg,
        on="id_configuracion",
        how="left"
    )

    df["year"] = df["fc_instalacion"].dt.year
    df["week"] = df["fc_instalacion"].dt.isocalendar().week.astype(int)

    df["year_week"] = (
        df["year"].astype(str)
        + "_"
        + df["week"].astype(str)
    )

    df = df[
        [
            "id_sitio",
            "nombre_sitio",
            "region",
            "id_configuracion",
            "altura_antena",
            "fc_instalacion",
            "year",
            "week",
            "year_week",
            "id_referencia",
            "referencia",
            "unidad",
            "cantidad_proyectada",
            "variacion_pasos",
            "rango_min",
            "rango_max",
        ]
    ].copy()

    return df

#%%
# Agregacion semanal de la demanda proyectada por referencia

def construir_demanda_proyectada_semanal(
    df_sitios_x_configuracion: pd.DataFrame
) -> pd.DataFrame:
    """
    Agrupa la demanda proyectada por semana y referencia.
    """
    df = df_sitios_x_configuracion.copy()

    df_demanda = (
        df.groupby(["year_week", "year", "week", "id_referencia", "referencia", "unidad"], as_index=False)
          .agg(cantidad_proyectada=("cantidad_proyectada", "sum"))
          .sort_values(["year_week", "id_referencia"])
          .reset_index(drop=True)
    )

    return df_demanda

#%%
# Pedidos de hardware a partir de la demanda proyectada semanal

def _year_week_a_lunes(year_week: str) -> pd.Timestamp:
    """
    Convierte un year_week con formato yyyy_ww a la fecha del lunes de esa semana iso.
    """
    year_str, week_str = year_week.split("_")
    year = int(year_str)
    week = int(week_str)
    return pd.Timestamp.fromisocalendar(year, week, 1)


def _lunes_a_year_week(fecha_lunes: pd.Timestamp) -> str:
    """
    Convierte la fecha del lunes de una semana iso al formato yyyy_ww.
    """
    iso = fecha_lunes.isocalendar()
    return f"{int(iso.year)}_{int(iso.week)}"



#%%
# Pedidos por bloques a partir de la demanda proyectada semanal, incluyendo pedido inicial

def construir_pedidos_desde_proyeccion(
    df_demanda_proyectada_semanal: pd.DataFrame,
    lead_weeks: int = 6,
    ciclo_weeks: int = 6,
    cobertura_weeks: int = 6
) -> pd.DataFrame:
    """
    Construye pedidos por bloques.

    Se incluye un pedido inicial antes del inicio del proyecto para dejar disponible HW en la semana 1,
    cubriendo semanas 1 a 6. Luego se generan pedidos por bloques:
    - Pedido semana 1  -> disponible semana 7  -> cubre semanas 7 a 12
    - Pedido semana 7  -> disponible semana 13 -> cubre semanas 13 a 18
    - Pedido semana 13 -> disponible semana 19 -> cubre semanas 19 a 24
    Y asi sucesivamente hasta cubrir todo el horizonte de la demanda proyectada.
    """
    df = df_demanda_proyectada_semanal.copy()

    df["fc_week_lunes"] = df["year_week"].apply(_year_week_a_lunes)

    semana_min = df["fc_week_lunes"].min()
    semana_max = df["fc_week_lunes"].max()

    semanas_calendario = pd.date_range(
        start=semana_min,
        end=semana_max,
        freq="W-MON"
    )

    pedidos = []

    fc_disponible_inicial = semanas_calendario[0]
    fc_pedido_inicial = fc_disponible_inicial - pd.to_timedelta(lead_weeks * 7, unit="D")
    fc_fin_ventana_inicial = fc_disponible_inicial + pd.to_timedelta((cobertura_weeks - 1) * 7, unit="D")

    df_ventana_inicial = df[
        (df["fc_week_lunes"] >= fc_disponible_inicial) &
        (df["fc_week_lunes"] <= fc_fin_ventana_inicial)
    ].copy()

    if not df_ventana_inicial.empty:
        df_pedido_inicial = (
            df_ventana_inicial
            .groupby(["id_referencia", "referencia", "unidad"], as_index=False)
            .agg(cantidad_pedido=("cantidad_proyectada", "sum"))
        )

        df_pedido_inicial["year_week_pedido"] = _lunes_a_year_week(fc_pedido_inicial)
        df_pedido_inicial["year_week_disponible"] = _lunes_a_year_week(fc_disponible_inicial)

        df_pedido_inicial = df_pedido_inicial[
            [
                "year_week_pedido",
                "year_week_disponible",
                "id_referencia",
                "referencia",
                "unidad",
                "cantidad_pedido"
            ]
        ].copy()

        pedidos.append(df_pedido_inicial)

    idx_pedido = np.arange(0, len(semanas_calendario), ciclo_weeks)
    semanas_pedido = semanas_calendario[idx_pedido]

    for fc_pedido_lunes in semanas_pedido:

        fc_disponible_lunes = fc_pedido_lunes + pd.to_timedelta(lead_weeks * 7, unit="D")
        fc_fin_ventana = fc_disponible_lunes + pd.to_timedelta((cobertura_weeks - 1) * 7, unit="D")

        df_ventana = df[
            (df["fc_week_lunes"] >= fc_disponible_lunes) &
            (df["fc_week_lunes"] <= fc_fin_ventana)
        ].copy()

        if df_ventana.empty:
            continue

        df_pedido = (
            df_ventana
            .groupby(["id_referencia", "referencia", "unidad"], as_index=False)
            .agg(cantidad_pedido=("cantidad_proyectada", "sum"))
        )

        df_pedido["year_week_pedido"] = _lunes_a_year_week(fc_pedido_lunes)
        df_pedido["year_week_disponible"] = _lunes_a_year_week(fc_disponible_lunes)

        df_pedido = df_pedido[
            [
                "year_week_pedido",
                "year_week_disponible",
                "id_referencia",
                "referencia",
                "unidad",
                "cantidad_pedido"
            ]
        ].copy()

        pedidos.append(df_pedido)

    if not pedidos:
        return pd.DataFrame(
            columns=[
                "year_week_pedido",
                "year_week_disponible",
                "id_referencia",
                "referencia",
                "unidad",
                "cantidad_pedido"
            ]
        )

    df_pedidos = pd.concat(pedidos, ignore_index=True)

    df_pedidos["cantidad_pedido"] = df_pedidos["cantidad_pedido"].astype(float)

    return df_pedidos


#%%
# Contaminacion de la cantidad proyectada para simular cantidad real

def _redondear_a_pasos(valor: float, paso: float) -> float:
    """
    Redondea un valor al multiplo mas cercano del paso definido.
    """
    if paso == 0:
        return float(valor)
    return float(np.round(valor / paso) * paso)


def _delta_intensidad_discreto(rango_min: float, rango_max: float, paso: float, intensidad: float) -> float:
    """
    Genera un delta discreto controlado por intensidad, privilegiando valores cerca de 0.
    """
    intensidad = float(np.clip(intensidad, 0.0, 1.0))

    if rango_min > 0 or rango_max < 0:
        delta = np.random.uniform(rango_min, rango_max)
        return _redondear_a_pasos(delta, paso)

    max_neg = abs(rango_min)
    max_pos = abs(rango_max)

    if max_neg == 0 and max_pos == 0:
        return 0.0

    if max_neg > 0 and max_pos > 0:
        signo = np.random.choice([-1, 1])
    elif max_neg > 0:
        signo = -1
    else:
        signo = 1

    if signo == -1:
        limite = max_neg * intensidad
        delta = -np.random.uniform(0, limite)
    else:
        limite = max_pos * intensidad
        delta = np.random.uniform(0, limite)

    return _redondear_a_pasos(delta, paso)


def contaminar_cantidad_real_por_sitio(
    df_sitios_x_configuracion: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 5,
    intensidad: float = 0.6,
    p_extremos: float = 0.15
) -> pd.DataFrame:
    """
    Contamina la cantidad proyectada para simular cantidad real.

    Por cada sitio se selecciona aleatoriamente un numero k de referencias a contaminar.
    Para cada referencia contaminada se calcula un delta en el rango permitido con pasos definidos,
    controlando la magnitud con intensidad y dejando una fraccion de eventos extremos.
    """
    df = df_sitios_x_configuracion.copy()

    campos_requeridos = ["id_sitio", "cantidad_proyectada", "variacion_pasos", "rango_min", "rango_max"]
    faltantes = [c for c in campos_requeridos if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan campos requeridos en el dataframe: {', '.join(faltantes)}")

    df["cantidad_real"] = df["cantidad_proyectada"].astype(float)
    df["fue_contaminada"] = 0

    for id_sitio, df_site in df.groupby("id_sitio", sort=False):
        idx_site = df_site.index.to_numpy()
        n_refs = len(idx_site)

        if n_refs == 0:
            continue

        k = int(np.random.randint(k_min, k_max + 1))
        k = min(k, n_refs)

        idx_sel = np.random.choice(idx_site, size=k, replace=False)

        for idx in idx_sel:
            rmin = float(df.at[idx, "rango_min"])
            rmax = float(df.at[idx, "rango_max"])
            paso = float(df.at[idx, "variacion_pasos"])
            base = float(df.at[idx, "cantidad_proyectada"])

            if np.random.rand() < float(np.clip(p_extremos, 0.0, 1.0)):
                delta = np.random.uniform(rmin, rmax)
                delta = _redondear_a_pasos(delta, paso)
            else:
                delta = _delta_intensidad_discreto(rmin, rmax, paso, intensidad)

            cantidad_real = base + delta
            if cantidad_real < 0:
                cantidad_real = 0.0

            df.at[idx, "cantidad_real"] = cantidad_real
            df.at[idx, "fue_contaminada"] = 1

    if "unidad" in df.columns:
        mask_und = df["unidad"].astype(str).str.lower().eq("und")
        df.loc[mask_und, "cantidad_real"] = df.loc[mask_und, "cantidad_real"].round(0).astype(int)
        df.loc[~mask_und, "cantidad_real"] = df.loc[~mask_und, "cantidad_real"].astype(float)

    return df




#%%
# Ejecucion del paso 01

if __name__ == "__main__":

    np.random.seed(42)

    ruta_config_hw = "01_simulation/input/configuracion_hw.xlsx"
    ruta_sitios = "01_simulation/input/sitios_proyecto.xlsx"

    # Carga configuracion base de hardware
    df_config_hw = cargar_configuracion_hw(ruta_config_hw)

    # Carga listado base de sitios del proyecto
    df_sitios = cargar_sitios_proyecto(ruta_sitios)

    # Asignacion de configuracion por sitio
    df_sitios = asignar_id_configuracion(df_sitios)

    # Asignacion de altura de antena por sitio
    df_sitios = asignar_altura_antena(df_sitios)

    # Asignacion de fechas de instalacion por sitio
    df_sitios = asignar_fc_instalacion_mensual(df_sitios)

    # Construccion del detalle sitio por referencia
    df_sitios_x_configuracion = construir_sitios_x_configuracion(df_sitios, df_config_hw)

    # Agregacion semanal de demanda proyectada
    df_demanda_proyectada_semanal = construir_demanda_proyectada_semanal(df_sitios_x_configuracion)

    # Construccion del plan de pedidos de hardware
    df_pedidos = construir_pedidos_desde_proyeccion(
        df_demanda_proyectada_semanal,
        lead_weeks=6,
        cobertura_weeks=7
    )
    
    # Ejecucion de la contaminacion para cantidad real

    df_sitios_x_configuracion_real = contaminar_cantidad_real_por_sitio(
        df_sitios_x_configuracion,
        k_min=2,
        k_max=5,
        intensidad=0.6,
        p_extremos=0.15
    )



#%%
# Salida para revision en excel

    ruta_salida = "01_simulation/output/sitios_proyecto_enriquecido.xlsx"
    os.makedirs("01_simulation/output", exist_ok=True)

    df_sitios.to_excel(
        ruta_salida,
        index=False,
        sheet_name="sitios_proyecto"
    )

    print(f"\nArchivo exportado en {ruta_salida}")
    
#%%
# Salida para revision del detalle sitio por referencia

    ruta_salida_detalle = "01_simulation/output/sitios_x_configuracion.xlsx"
    os.makedirs("01_simulation/output", exist_ok=True)

    df_sitios_x_configuracion.to_excel(
        ruta_salida_detalle,
        index=False,
        sheet_name="sitios_x_configuracion"
    )

    print(f"\nArchivo exportado en {ruta_salida_detalle}")
    
#%%
# Salida para revision de la demanda proyectada semanal

    ruta_salida_demanda = "01_simulation/output/demanda_proyectada_semanal.xlsx"
    os.makedirs("01_simulation/output", exist_ok=True)

    df_demanda_proyectada_semanal.to_excel(
        ruta_salida_demanda,
        index=False,
        sheet_name="demanda_proyectada_semanal"
    )

    print(f"\nArchivo exportado en {ruta_salida_demanda}")
    
#%%
# Salida para revision del plan de pedidos

    ruta_salida_pedidos = "01_simulation/output/pedidos_desde_proyeccion.xlsx"
    os.makedirs("01_simulation/output", exist_ok=True)

    df_pedidos.to_excel(
        ruta_salida_pedidos,
        index=False,
        sheet_name="pedidos"
    )

    print(f"\nArchivo exportado en {ruta_salida_pedidos}")
    
    
#%%
# Salida para revision del detalle con cantidad real

    ruta_salida_real = "01_simulation/output/sitios_x_configuracion_con_real.xlsx"
    os.makedirs("01_simulation/output", exist_ok=True)

    df_sitios_x_configuracion_real.to_excel(
        ruta_salida_real,
        index=False,
        sheet_name="sitios_x_configuracion_real"
    )

    print(f"\nArchivo exportado en {ruta_salida_real}")



