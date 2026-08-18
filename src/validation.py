"""
Validación — split temporal y comparación k-fold aleatorio vs out-of-time.

Objetivo de este módulo:
  1. Proveer un split temporal reutilizable (se usa también en train.py).
  2. Demostrar, con números, por qué el k-fold aleatorio da una estimación optimista
     frente a la validación out-of-time (OOT), que es la honesta para este problema.

Uso:
    python src/validation.py
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

from data_prep import GOLD_DIR, COL_ID, COL_TARGET, COL_FECHA

FECHA_CORTE_OOT = "2024-12-01"  # val = últimos ~3 meses del train (mimetiza el test)
# Cortes para el split de tres vías (train / val / holdout), todos temporales:
FECHA_CORTE_VAL = "2024-10-01"      # train < corte_val
FECHA_CORTE_HOLDOUT = "2024-12-01"  # val en [corte_val, corte_holdout); holdout >= corte_holdout
SEED = 42


# --- Split temporal ----------------------------------------------------------
def split_temporal(df: pd.DataFrame, fecha_corte: str = FECHA_CORTE_OOT):
    """Divide por fecha: antes del corte = train; desde el corte = validación OOT."""
    corte = pd.Timestamp(fecha_corte)
    fechas = pd.to_datetime(df[COL_FECHA])
    tr = df[fechas < corte].copy()
    va = df[fechas >= corte].copy()
    return tr, va


def split_temporal_3(df: pd.DataFrame,
                     corte_val: str = FECHA_CORTE_VAL,
                     corte_holdout: str = FECHA_CORTE_HOLDOUT):
    """Split temporal de tres vías: train (tunea) / val (elige) / holdout (reporta 1 vez).

    Todo temporal, para imitar el salto real train->test y evitar contaminar la
    estimación honesta al tunear.
    """
    fechas = pd.to_datetime(df[COL_FECHA])
    cv, ch = pd.Timestamp(corte_val), pd.Timestamp(corte_holdout)
    train = df[fechas < cv].copy()
    val = df[(fechas >= cv) & (fechas < ch)].copy()
    holdout = df[fechas >= ch].copy()
    return train, val, holdout


# --- Columnas de features ----------------------------------------------------
def columnas_num_cat(df: pd.DataFrame):
    """Separa columnas numéricas y categóricas (excluye id, target y fecha).

    Se usa is_numeric_dtype (robusto) en vez de comparar dtype == 'object', que
    falla si el parquet devuelve las strings como dtype 'string' en vez de 'object'.
    """
    excluir = {COL_ID, COL_TARGET, COL_FECHA}
    feats = [c for c in df.columns if c not in excluir]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in feats if c not in num]
    return num, cat


# --- Pipeline del baseline (regresión logística) -----------------------------
def build_baseline_pipeline(num, cat):
    """Logística con imputación + escalado (num) y one-hot (cat), ajustado en train.

    class_weight='balanced' maneja el desbalance (~10% default) sin resamplear.
    Toda transformación con estado vive AQUÍ y se ajusta solo con el train de cada
    fold => sin leakage.
    """
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def ks_score(y_true, y_score):
    """KS = máxima separación entre las acumuladas de buenos y malos."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def main():
    gold = pd.read_parquet(GOLD_DIR / "train_gold.parquet")
    num, cat = columnas_num_cat(gold)
    X, y = gold[num + cat], gold[COL_TARGET]
    print(f"Features: {len(num)} numéricas + {len(cat)} categóricas | filas: {len(gold)}")

    # --- (1) k-fold ALEATORIO: mezcla periodos => estimación OPTIMISTA -----------
    pipe = build_baseline_pipeline(num, cat)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    auc_kfold = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print("\n(1) k-fold aleatorio (5 folds):")
    print(f"    AUC = {auc_kfold.mean():.4f} ± {auc_kfold.std():.4f}")

    # --- (2) OUT-OF-TIME: entrena en el pasado, valida en el futuro (HONESTO) ----
    tr, va = split_temporal(gold)
    pipe_oot = build_baseline_pipeline(num, cat)
    pipe_oot.fit(tr[num + cat], tr[COL_TARGET])
    p_va = pipe_oot.predict_proba(va[num + cat])[:, 1]

    auc_oot = roc_auc_score(va[COL_TARGET], p_va)
    ks_oot = ks_score(va[COL_TARGET], p_va)
    print(f"\n(2) Out-of-time (train < {FECHA_CORTE_OOT} | val >= {FECHA_CORTE_OOT}):")
    print(f"    train={len(tr)}  val={len(va)}")
    print(f"    AUC = {auc_oot:.4f}   KS = {ks_oot:.4f}")

    # --- Calibración: ¿la probabilidad media predicha coincide con la real? -------
    print(f"\n    Calibración en la ventana OOT:")
    print(f"      default real en val   : {va[COL_TARGET].mean()*100:.2f}%")
    print(f"      prob. media predicha  : {p_va.mean()*100:.2f}%")

    print("\n--- Lectura ---")
    print(f"    Brecha AUC (kfold - OOT) = {auc_kfold.mean()-auc_oot:+.4f}")


if __name__ == "__main__":
    main()