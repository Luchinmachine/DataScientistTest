"""
Entrenamiento y comparación de modelos (champion-challenger).

Compara tres familias sobre el MISMO holdout out-of-time, con las mismas métricas
(discriminación: AUC/KS; calibración: prob. media y Brier). Usa el set honesto de
features (la variable con leakage queda excluida por defecto en columnas_num_cat).

  - Regresión logística  (lineal, interpretable, estándar de scorecards)
  - Random Forest        (bagging)
  - XGBoost              (boosting; maneja NaN nativo)

Diseño anti-leakage: toda transformación con estado (imputación, one-hot, escalado)
va dentro del pipeline y se ajusta SOLO con el train.

Uso:
    python src/train.py
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

from data_prep import GOLD_DIR, COL_TARGET
from validation import split_temporal_3, columnas_num_cat, ks_score

SEED = 42


def _onehot():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_pipelines(num, cat):
    """Un pipeline por familia, cada uno con el preprocesamiento que necesita."""
    # Logística: imputa + escala numéricas, one-hot categóricas.
    pre_log = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", _onehot(), cat)])
    # Random Forest: imputa numéricas (no necesita escalar), one-hot categóricas.
    pre_rf = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", _onehot(), cat)])
    # XGBoost: NO imputa (maneja NaN nativo), one-hot categóricas.
    pre_xgb = ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", _onehot(), cat)])

    return {
        "LogReg": Pipeline([("pre", pre_log),
                            ("clf", LogisticRegression(max_iter=1000))]),
        "RandomForest": Pipeline([("pre", pre_rf),
                                  ("clf", RandomForestClassifier(
                                      n_estimators=300, max_depth=8,
                                      min_samples_leaf=50, random_state=SEED, n_jobs=-1))]),
        "XGBoost": Pipeline([("pre", pre_xgb),
                             ("clf", XGBClassifier(
                                 n_estimators=300, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="logloss", random_state=SEED))]),
    }


def evaluar(pipe, cols, tr, ev):
    """Ajusta en tr y evalúa en ev: AUC, KS, prob media predicha y Brier."""
    pipe.fit(tr[cols], tr[COL_TARGET])
    p = pipe.predict_proba(ev[cols])[:, 1]
    y = ev[COL_TARGET]
    return dict(auc=roc_auc_score(y, p), ks=ks_score(y, p),
                prob_media=p.mean(), brier=brier_score_loss(y, p))


def main():
    gold = pd.read_parquet(GOLD_DIR / "train_gold.parquet")
    num, cat = columnas_num_cat(gold)           # set honesto (sin la variable con leakage)
    cols = num + cat
    train, val, holdout = split_temporal_3(gold)
    print(f"Features: {len(num)} num + {len(cat)} cat | "
          f"train={len(train)} val={len(val)} holdout={len(holdout)}")
    print(f"default holdout: {holdout[COL_TARGET].mean()*100:.2f}%\n")

    pipes = build_pipelines(num, cat)

    # Selección: entrena en train, mide en VAL (para elegir sin tocar el holdout).
    print(f"{'modelo':13s} {'AUC_val':>8s} {'KS_val':>7s} {'AUC_hold':>9s} "
          f"{'KS_hold':>8s} {'prob_med':>9s} {'Brier':>7s}")
    resultados = {}
    for nombre, pipe in pipes.items():
        rv = evaluar(pipe, cols, train, val)
        rh = evaluar(pipe, cols, train, holdout)   # holdout: reporte final
        resultados[nombre] = rh
        print(f"{nombre:13s} {rv['auc']:>8.4f} {rv['ks']:>7.4f} "
              f"{rh['auc']:>9.4f} {rh['ks']:>8.4f} {rh['prob_media']*100:>8.2f}% {rh['brier']:>7.4f}")

    print(f"\n(referencia: default real holdout = {holdout[COL_TARGET].mean()*100:.2f}%)")


if __name__ == "__main__":
    main()
