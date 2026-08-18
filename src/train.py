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
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
# pyrefly: ignore [missing-import]
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


def _metricas(y, p):
    return dict(auc=roc_auc_score(y, p), ks=ks_score(y, p),
                prob=p.mean(), brier=brier_score_loss(y, p))


def main():
    gold = pd.read_parquet(GOLD_DIR / "train_gold.parquet")
    num, cat = columnas_num_cat(gold)           # set honesto (sin la variable con leakage)
    cols = num + cat
    train, val, holdout = split_temporal_3(gold)
    print(f"Features: {len(num)} num + {len(cat)} cat | "
          f"train={len(train)} val={len(val)} holdout={len(holdout)}")
    print(f"default holdout: {holdout[COL_TARGET].mean()*100:.2f}%\n")

    pipes = build_pipelines(num, cat)

    # Ajusta cada modelo en train; guarda predicciones en val y holdout.
    val_preds, hold_preds = {}, {}
    for nombre, pipe in pipes.items():
        pipe.fit(train[cols], train[COL_TARGET])
        val_preds[nombre] = pipe.predict_proba(val[cols])[:, 1]
        hold_preds[nombre] = pipe.predict_proba(holdout[cols])[:, 1]

    # Ensemble (promedio simple). Se documenta que se probó; ver diversidad abajo.
    val_preds["Ensemble(mean)"] = np.mean([val_preds[n] for n in pipes], axis=0)
    hold_preds["Ensemble(mean)"] = np.mean([hold_preds[n] for n in pipes], axis=0)

    yv, yh = val[COL_TARGET], holdout[COL_TARGET]
    print(f"{'modelo':15s} {'AUC_val':>8s} {'KS_val':>7s} {'AUC_hold':>9s} "
          f"{'KS_hold':>8s} {'prob_med':>9s} {'Brier':>7s}")
    for nombre in hold_preds:
        rv, rh = _metricas(yv, val_preds[nombre]), _metricas(yh, hold_preds[nombre])
        print(f"{nombre:15s} {rv['auc']:>8.4f} {rv['ks']:>7.4f} "
              f"{rh['auc']:>9.4f} {rh['ks']:>8.4f} {rh['prob']*100:>8.2f}% {rh['brier']:>7.4f}")

    # Diversidad: correlación media entre las predicciones de los modelos base.
    base = [hold_preds[n] for n in pipes]
    corrs = [spearmanr(base[i], base[j]).correlation
             for i in range(len(base)) for j in range(i + 1, len(base))]
    print(f"\nCorrelación media entre modelos base: {np.mean(corrs):.3f} "
          f"(alta => ensemble aporta poco)")
    print(f"(referencia: default real holdout = {yh.mean()*100:.2f}%)")


if __name__ == "__main__":
    main()