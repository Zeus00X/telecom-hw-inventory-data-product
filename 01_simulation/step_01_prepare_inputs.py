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
    Construye el detalle de hardware por sitio, replicando las referencias
    segun la configuracion asignada y derivando la dimension temporal semanal.
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
# Ejecucion del paso 01

if __name__ == "__main__":

    np.random.seed(42)

    ruta_config_hw = "01_simulation/input/configuracion_hw.xlsx"
    ruta_sitios = "01_simulation/input/sitios_proyecto.xlsx"

    df_config_hw = cargar_configuracion_hw(ruta_config_hw)
    df_sitios = cargar_sitios_proyecto(ruta_sitios)

    df_sitios = asignar_id_configuracion(df_sitios)
    df_sitios = asignar_altura_antena(df_sitios)
    df_sitios = asignar_fc_instalacion_mensual(df_sitios)
    
    df_sitios_x_configuracion = construir_sitios_x_configuracion(df_sitios, df_config_hw)

    df_demanda_proyectada_semanal = construir_demanda_proyectada_semanal(df_sitios_x_configuracion)



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

