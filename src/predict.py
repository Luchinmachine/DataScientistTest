"""
Modelo final y generación de predictions.csv.

Terminada la validación, el modelo de producción usa TODA la data disponible, repartida
con criterio temporal para no desperdiciar nada y calibrar sobre el régimen más reciente:

  - Modelo (logística):   se entrena con train < CALIB_CUTOFF (la mayor parte del train).
  - Calibrador (isotónica): se ajusta con train >= CALIB_CUTOFF (el tramo más reciente,
                            el más parecido al régimen del test).

Genera predictions.csv (id_solicitud, prob_default) en la raíz del repo.

Uso:
    python src/predict.py
"""

import pandas as pd
from sklearn.isotonic import IsotonicRegression

from data_prep import GOLD_DIR, PROJECT_ROOT, COL_ID, COL_TARGET, COL_FECHA
from validation import columnas_num_cat
from train import build_pipelines

CALIB_CUTOFF = "2025-01-01"   # modelo: train < corte; calibrador: train >= corte


def main():
    train = pd.read_parquet(GOLD_DIR / "train_gold.parquet")
    test = pd.read_parquet(GOLD_DIR / "test_gold.parquet")
    num, cat = columnas_num_cat(train)          # set honesto (sin la variable con leakage)
    cols = num + cat

    # Reparto temporal de TODA la data: fit del modelo + calibración reciente.
    fechas = pd.to_datetime(train[COL_FECHA])
    corte = pd.Timestamp(CALIB_CUTOFF)
    fit = train[fechas < corte]
    cal = train[fechas >= corte]
    print(f"Modelo entrenado con {len(fit)} filas (< {CALIB_CUTOFF}); "
          f"calibrado con {len(cal)} filas (>= {CALIB_CUTOFF}).")

    # Modelo campeón + calibración isotónica (sobre el tramo reciente).
    base = build_pipelines(num, cat)["LogReg"]
    base.fit(fit[cols], fit[COL_TARGET])
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        base.predict_proba(cal[cols])[:, 1], cal[COL_TARGET])

    # Predicción calibrada sobre el test.
    prob = iso.predict(base.predict_proba(test[cols])[:, 1])

    out = pd.DataFrame({COL_ID: test[COL_ID].values, "prob_default": prob})
    ruta = PROJECT_ROOT / "predictions.csv"
    out.to_csv(ruta, index=False)

    # Chequeos de sanidad del entregable.
    print(f"\npredictions.csv -> {ruta}")
    print(f"  filas: {len(out)} (debe ser 12000)")
    print(f"  columnas: {list(out.columns)}")
    print(f"  prob_default en [0,1]: {out['prob_default'].between(0,1).all()}")
    print(f"  prob media predicha: {out['prob_default'].mean()*100:.2f}% "
          f"(referencia: default reciente del train ~12-13%)")
    print(f"  ids únicos: {out[COL_ID].is_unique}  | rango: "
          f"{out[COL_ID].min()}-{out[COL_ID].max()}")


if __name__ == "__main__":
    main()
