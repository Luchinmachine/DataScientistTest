"""
Limpieza de datos.

Transformaciones SIN ESTADO (deterministas, fila a fila): se aplican idénticas a
train y test, no aprenden nada de la distribución. Implementan las decisiones
documentadas en notebooks/01_eda.ipynb.

Las transformaciones CON ESTADO (imputación, codificación, escalado) NO van aquí:
aprenden parámetros de los datos y deben ajustarse solo con train para evitar
leakage. Viven en el pipeline del modelo (src/train.py).

Uso:
    from data_prep import clean
    train = clean(pd.read_csv("data/train.csv"))
"""

import numpy as np
import pandas as pd

# --- Umbrales documentados (ver EDA) -----------------------------------------
INGRESO_UMBRAL_ERROR = 10_000      # ingresos por debajo => error de unidad
INGRESO_FACTOR = 1_000             # corrección: estaban en miles, no en pesos
EDAD_MAX_PLAUSIBLE = 100           # edades por encima => error de captura
ANTIG_LABORAL_MAX_MESES = 600      # 50 años de antigüedad laboral => imposible

# Columnas que no son features (identificador, target, fecha para el split)
COL_ID = "id_solicitud"
COL_TARGET = "default_12m"
COL_FECHA = "fecha_solicitud"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la limpieza determinística. Devuelve un DataFrame nuevo (no muta)."""
    df = df.copy()

    # 1. Corrección de ingreso (error de unidad, validado por reconciliación ×1000).
    #    Se marca con un indicador como seguro ante correcciones imperfectas.
    mask_err = df["ingreso_declarado"].notna() & (df["ingreso_declarado"] < INGRESO_UMBRAL_ERROR)
    df["ingreso_fue_corregido"] = mask_err.astype(int)
    df.loc[mask_err, "ingreso_declarado"] = df.loc[mask_err, "ingreso_declarado"] * INGRESO_FACTOR

    # 2. Valores imposibles -> NaN (impacto marginal; se documentan).
    df.loc[df["edad"] > EDAD_MAX_PLAUSIBLE, "edad"] = np.nan
    df.loc[df["antiguedad_laboral_meses"] > ANTIG_LABORAL_MAX_MESES,
           "antiguedad_laboral_meses"] = np.nan

    # 3. Indicadores de faltante (el faltante NO es aleatorio => puede ser predictivo).
    #    Se crean DESPUÉS de marcar imposibles como NaN, para capturarlos también.
    for col in ["ingreso_declarado", "antiguedad_laboral_meses"]:
        df[f"{col}_faltante"] = df[col].isna().astype(int)

    # 4. Fecha a datetime (necesaria para el split temporal; no se usa como feature cruda).
    df[COL_FECHA] = pd.to_datetime(df[COL_FECHA])

    return df


def columnas_feature(df: pd.DataFrame) -> list[str]:
    """Columnas usables como predictores (excluye id, target y fecha)."""
    excluir = {COL_ID, COL_TARGET, COL_FECHA}
    return [c for c in df.columns if c not in excluir]


if __name__ == "__main__":
    # Chequeo de sanidad: aplica clean() a train y test y verifica el resultado.
    from pathlib import Path
    DATA = Path("data")

    for nombre in ["train", "test"]:
        raw = pd.read_csv(DATA / f"{nombre}.csv")
        out = clean(raw)
        print(f"\n=== {nombre} ===")
        print(f"  shape: {raw.shape} -> {out.shape}")
        print(f"  ingresos corregidos: {int(out['ingreso_fue_corregido'].sum())}")
        print(f"  ingreso < {INGRESO_UMBRAL_ERROR} tras corrección: "
              f"{int((out['ingreso_declarado'] < INGRESO_UMBRAL_ERROR).sum())} (debe ser 0)")
        print(f"  edades > {EDAD_MAX_PLAUSIBLE} tras limpieza: "
              f"{int((out['edad'] > EDAD_MAX_PLAUSIBLE).sum())} (debe ser 0)")
        print(f"  nuevas columnas: "
              f"{sorted(set(out.columns) - set(raw.columns))}")
