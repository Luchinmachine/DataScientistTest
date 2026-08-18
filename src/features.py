"""
Features de negocio — capa silver -> gold.

Transformaciones SIN ESTADO: derivan features fila a fila a partir del silver.
NO incluye codificación/imputación/escalado (transformaciones con estado que se
ajustan solo con train; viven en el pipeline del modelo, src/train.py).

Uso:
    python src/features.py          # regenera data/gold/*_gold.parquet
"""

import numpy as np
import pandas as pd

from data_prep import SILVER_DIR, GOLD_DIR


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features de negocio al silver. Devuelve un DataFrame nuevo."""
    df = df.copy()

    # Ratio deuda/ingreso (debt-to-income): carga de deuda existente vs ingreso.
    #   Clásico de riesgo: a mayor DTI, mayor probabilidad de mora.
    df["ratio_deuda_ingreso"] = df["deuda_sistema"] / df["ingreso_declarado"]

    # Ratio monto/ingreso (loan-to-income): tamaño del crédito pedido vs ingreso mensual.
    df["ratio_monto_ingreso"] = df["monto_solicitado"] / df["ingreso_declarado"]

    # Nota: la división propaga NaN si el ingreso es nulo (no se imputa aquí);
    # el ingreso mínimo tras la corrección es > 100.000, así que no hay división por cero.
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def procesar_gold(nombre: str) -> pd.DataFrame:
    """Lee la capa silver, construye features y materializa la capa gold."""
    silver = pd.read_parquet(SILVER_DIR / f"{nombre}_silver.parquet")
    gold = build_features(silver)

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    ruta = GOLD_DIR / f"{nombre}_gold.parquet"
    gold.to_parquet(ruta, index=False)

    nuevas = sorted(set(gold.columns) - set(silver.columns))
    print(f"=== {nombre} ===  {silver.shape} -> {gold.shape}")
    print(f"  features nuevas: {nuevas}")
    print(f"  gold -> {ruta}")
    return gold


if __name__ == "__main__":
    for nombre in ["train", "test"]:
        procesar_gold(nombre)
