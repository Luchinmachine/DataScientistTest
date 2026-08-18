"""
Calibración del modelo campeón (regresión logística).

El modelo discrimina bien pero subpredice (drift temporal: entrenado en un período
de menor default). Se calibra para que la probabilidad refleje el riesgo real, porque
la política de aprobación usa la probabilidad en un cálculo de valor esperado.

Protocolo (sin leakage): modelo ajustado en TRAIN, calibrador ajustado en VALIDACIÓN,
evaluación en HOLDOUT. La calibración es monótona => AUC y KS no cambian; sí mejoran
la calibración-en-el-nivel, el ECE y el Brier.

Uso:
    python src/calibracion.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

from data_prep import GOLD_DIR, COL_TARGET, PROJECT_ROOT
from validation import split_temporal_3, columnas_num_cat, ks_score
from train import build_pipelines

FIG = PROJECT_ROOT / "reports" / "figures"


def ece(y, p, bins=10):
    """Expected Calibration Error con bins por cuantiles (robusto a scores sesgados)."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi)
        if m.sum() > 0:
            total += abs(p[m].mean() - y[m].mean()) * m.sum() / len(y)
    return total


def resumen(nombre, y, p):
    print(f"  {nombre:16s} AUC={roc_auc_score(y,p):.4f}  KS={ks_score(y,p):.4f}  "
          f"prob_media={p.mean()*100:5.2f}%  ECE={ece(y,p):.4f}  Brier={brier_score_loss(y,p):.4f}")


def main():
    gold = pd.read_parquet(GOLD_DIR / "train_gold.parquet")
    num, cat = columnas_num_cat(gold); cols = num + cat
    train, val, holdout = split_temporal_3(gold)

    # Modelo campeón (logística), ajustado SOLO en train.
    base = build_pipelines(num, cat)["LogReg"]
    base.fit(train[cols], train[COL_TARGET])

    val_p = base.predict_proba(val[cols])[:, 1]
    hold_p = base.predict_proba(holdout[cols])[:, 1]
    yv, yh = val[COL_TARGET].values, holdout[COL_TARGET].values

    # Calibradores ajustados en VALIDACIÓN.
    platt = LogisticRegression().fit(val_p.reshape(-1, 1), yv)
    hold_platt = platt.predict_proba(hold_p.reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip").fit(val_p, yv)
    hold_iso = iso.predict(hold_p)

    print(f"default real en holdout: {yh.mean()*100:.2f}%\n")
    print("Evaluación en HOLDOUT (AUC/KS no cambian; mejora calibración, ECE, Brier):")
    resumen("Sin calibrar", yh, hold_p)
    resumen("Platt (sigmoide)", yh, hold_platt)
    resumen("Isotónica", yh, hold_iso)

    # Figura: curva de calibración (reliability diagram).
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Calibración perfecta")
    for nombre, p in [("Sin calibrar", hold_p), ("Platt", hold_platt), ("Isotónica", hold_iso)]:
        frac_pos, mean_pred = calibration_curve(yh, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=nombre)
    ax.set(xlabel="Probabilidad media predicha", ylabel="Default observado",
           title="Curva de calibración (holdout)")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "calibracion.png", dpi=120); plt.close(fig)
    print(f"\nFigura -> {FIG / 'calibracion.png'}")


if __name__ == "__main__":
    main()
