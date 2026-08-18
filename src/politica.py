"""
Política de aprobación basada en la economía del producto.

Regla: aprobar una solicitud si su valor esperado es positivo,
    EV = (1 - p) * G - p * L
donde p es la probabilidad de default CALIBRADA, G la ganancia si paga y L la pérdida
si cae en default. Equivale a aprobar cuando p < G/(G+L). Como G y L son proporcionales
al monto, el monto se cancela: el umbral depende solo del plazo.

Se estima la ganancia con un BACKTEST sobre el holdout etiquetado: se decide con la
probabilidad predicha (lo que tendrías al decidir) y se contabiliza la plata con el
resultado real (lo que efectivamente pasó).

Uso:
    python src/politica.py
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from data_prep import GOLD_DIR, COL_TARGET
from validation import split_temporal_3, columnas_num_cat
from train import build_pipelines

TASA_ANUAL = 0.12
FACTOR_AMORT = 0.5
LGD = 0.55


def economia(df):
    """Ganancia si paga (G) y pérdida si cae (L) por solicitud, según el enunciado."""
    G = df["monto_solicitado"] * TASA_ANUAL * (df["plazo_meses"] / 12) * FACTOR_AMORT
    L = df["monto_solicitado"] * LGD
    return G, L


def prob_calibrada(train, val, holdout, num, cat):
    """Modelo campeón (logística) + calibración isotónica; prob. calibrada en holdout."""
    cols = num + cat
    base = build_pipelines(num, cat)["LogReg"]
    base.fit(train[cols], train[COL_TARGET])
    val_p = base.predict_proba(val[cols])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(val_p, val[COL_TARGET])
    return iso.predict(base.predict_proba(holdout[cols])[:, 1])


def main():
    gold = pd.read_parquet(GOLD_DIR / "train_gold.parquet")
    num, cat = columnas_num_cat(gold)
    cols = num + cat
    train, val, holdout = split_temporal_3(gold)

    p = prob_calibrada(train, val, holdout, num, cat)
    G, L = economia(holdout)
    p_star = G / (G + L)                 # umbral por solicitud
    ev = (1 - p) * G.values - p * L.values
    aprobar = ev > 0
    y = holdout[COL_TARGET].values

    # Umbral por plazo (el monto se cancela => depende solo del plazo).
    print("Umbral de aprobación por plazo (p* = G/(G+L)):")
    tmp = pd.DataFrame({"plazo": holdout["plazo_meses"].values, "p_star": p_star.values})
    for plazo, g in sorted(tmp.groupby("plazo")):
        print(f"  plazo {int(plazo):>2} meses -> aprobar si prob_default < {g['p_star'].iloc[0]*100:5.2f}%")

    # Backtest: plata real (resultado observado) según la decisión.
    real = np.where(y == 0, G.values, -L.values)   # +G si pagó, -L si cayó
    gain_politica = real[aprobar].sum()
    gain_todo = real.sum()

    print("\n=== Backtest sobre el holdout (plata real) ===")
    print(f"  Solicitudes: {len(y)}  |  default real: {y.mean()*100:.2f}%")
    print(f"  Tasa de aprobación de la política: {aprobar.mean()*100:.1f}%")
    print(f"  Default entre APROBADOS: {y[aprobar].mean()*100:.2f}%  "
          f"(vs {y.mean()*100:.2f}% en toda la cartera)")
    print(f"\n  Ganancia APROBAR TODO : {gain_todo/1e6:>10.1f} MM CLP")
    print(f"  Ganancia POLÍTICA     : {gain_politica/1e6:>10.1f} MM CLP")
    print(f"  Mejora de la política : {(gain_politica-gain_todo)/1e6:>10.1f} MM CLP")
    print(f"\n  Ganancia por solicitud - aprobar todo: {gain_todo/len(y):>12,.0f} CLP")
    print(f"  Ganancia por solicitud - política     : {gain_politica/len(y):>12,.0f} CLP")


if __name__ == "__main__":
    main()
