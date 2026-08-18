"""
Comparación con vs sin `num_contactos_ult_trimestre` (variable con leakage sospechado).

Responde: ¿cuánto aporta la variable, y qué cambia al excluirla?

IMPORTANTE: el test NO tiene target, así que en test solo se compara el COMPORTAMIENTO
de la predicción (distribución, reordenamiento, tasa de aprobación), no la performance.
La performance (AUC/KS) se mide solo sobre datos etiquetados (train / ventana OOT).

Uso:
    python src/comparar_leakage.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

from data_prep import GOLD_DIR, COL_ID, COL_TARGET
from validation import (build_baseline_pipeline, split_temporal,
                        columnas_num_cat, ks_score, FECHA_CORTE_OOT)

VAR_SOSPECHOSA = "num_contactos_ult_trimestre"
UMBRAL_REF = 0.15  # umbral de referencia para comparar tasa de aprobación en test


def psi(esperado: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index: mide drift de una variable entre dos poblaciones."""
    e = esperado.dropna(); a = actual.dropna()
    cortes = np.quantile(e, np.linspace(0, 1, bins + 1))
    cortes[0], cortes[-1] = -np.inf, np.inf
    pe = np.clip(np.histogram(e, cortes)[0] / len(e), 1e-6, None)
    pa = np.clip(np.histogram(a, cortes)[0] / len(a), 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def main():
    train = pd.read_parquet(GOLD_DIR / "train_gold.parquet")
    test = pd.read_parquet(GOLD_DIR / "test_gold.parquet")
    num, cat = columnas_num_cat(train)
    num_sin = [c for c in num if c != VAR_SOSPECHOSA]

    tr, va = split_temporal(train)  # OOT: entrena en el pasado, valida en el futuro

    # =====================================================================
    # (A) PERFORMANCE en datos ETIQUETADOS (train interno vs OOT-val)
    # =====================================================================
    print("=" * 68)
    print("(A) PERFORMANCE con vs sin la variable (datos etiquetados)")
    print(f"    train={len(tr)}  val(OOT)={len(va)}  corte={FECHA_CORTE_OOT}\n")
    print(f"    {'variante':10s} {'AUC_train':>10s} {'AUC_val':>10s} {'KS_val':>8s} {'gap':>8s}")

    modelos = {}
    for etq, cols in [("CON", num + cat), ("SIN", num_sin + cat)]:
        pipe = build_baseline_pipeline([c for c in cols if c in num], cat)
        pipe.fit(tr[cols], tr[COL_TARGET])
        p_tr = pipe.predict_proba(tr[cols])[:, 1]
        p_va = pipe.predict_proba(va[cols])[:, 1]
        auc_tr = roc_auc_score(tr[COL_TARGET], p_tr)
        auc_va = roc_auc_score(va[COL_TARGET], p_va)
        ks_va = ks_score(va[COL_TARGET], p_va)
        print(f"    {etq:10s} {auc_tr:>10.4f} {auc_va:>10.4f} {ks_va:>8.4f} {auc_tr-auc_va:>+8.4f}")
        modelos[etq] = cols

    # =====================================================================
    # (B) COMPORTAMIENTO en TEST (sin etiquetas): distribución y reordenamiento
    # =====================================================================
    print("\n" + "=" * 68)
    print("(B) COMPORTAMIENTO en TEST (sin target: solo distribución/ranking)")

    preds_test = {}
    for etq, cols in [("CON", num + cat), ("SIN", num_sin + cat)]:
        pipe = build_baseline_pipeline([c for c in cols if c in num], cat)
        pipe.fit(train[cols], train[COL_TARGET])          # entrena con TODO el train
        preds_test[etq] = pipe.predict_proba(test[cols])[:, 1]

    print("\n    Distribución de prob_default predicha en test:")
    print(f"    {'variante':10s} {'media':>8s} {'p50':>8s} {'p90':>8s}")
    for etq in ["CON", "SIN"]:
        p = preds_test[etq]
        print(f"    {etq:10s} {p.mean():>8.3f} {np.median(p):>8.3f} {np.quantile(p,0.9):>8.3f}")

    rho = spearmanr(preds_test["CON"], preds_test["SIN"]).correlation
    print(f"\n    Correlación de ranking (Spearman) CON vs SIN: {rho:.4f}")
    print("      -> qué tanto reordena a los solicitantes al quitar la variable")

    for etq in ["CON", "SIN"]:
        aprob = (preds_test[etq] < UMBRAL_REF).mean()
        print(f"    Tasa de aprobación en test (umbral {UMBRAL_REF}) [{etq}]: {aprob*100:.1f}%")

    # =====================================================================
    # (C) CONTRIBUCIÓN de la variable (importancia por permutación en OOT-val)
    # =====================================================================
    print("\n" + "=" * 68)
    print("(C) CONTRIBUCIÓN de la variable (permutation importance, AUC en OOT-val)")
    cols = num + cat
    pipe = build_baseline_pipeline(num, cat)
    pipe.fit(tr[cols], tr[COL_TARGET])
    imp = permutation_importance(pipe, va[cols], va[COL_TARGET],
                                 scoring="roc_auc", n_repeats=5, random_state=42)
    ranking = (pd.Series(imp.importances_mean, index=cols)
               .sort_values(ascending=False).head(6))
    print("    Top-6 variables por caída de AUC al permutarlas:")
    for k, v in ranking.items():
        marca = "  <== sospechosa" if k == VAR_SOSPECHOSA else ""
        print(f"      {k:32s} {v:.4f}{marca}")

    # =====================================================================
    # (D) DRIFT de la variable entre train y test (PSI)
    # =====================================================================
    print("\n" + "=" * 68)
    print("(D) DRIFT train->test de la variable (PSI)")
    valor_psi = psi(train[VAR_SOSPECHOSA], test[VAR_SOSPECHOSA])
    guia = "sin drift" if valor_psi < 0.1 else ("drift moderado" if valor_psi < 0.25 else "drift alto")
    print(f"    PSI({VAR_SOSPECHOSA}) = {valor_psi:.4f}  ({guia})")


if __name__ == "__main__":
    main()