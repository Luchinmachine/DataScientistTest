# Modelo de Probabilidad de Default — Andina Crédito

Modelo que estima la probabilidad de *default* (mora 90+ días dentro de 12 meses del
desembolso) para créditos de consumo, y una **política de aprobación** basada en la
economía del producto. Desafío de Data Scientist.

## Resultado

- **Discriminación honesta:** AUC ≈ 0,83 · KS ≈ 0,51 (validación out-of-time).
- **Impacto de negocio:** la política casi cuadruplica la ganancia del portafolio
  (198 → 734 MM CLP en backtest) y reduce la mora entre aprobados de 12,5% a 6,1%.
- Ver `reports/informe_ejecutivo.md`.

## Estructura

```
.
├── data/
│   ├── raw/
│   │   ├── train.csv                              (versionado: insumo)
│   │   └── test.csv
│   ├── silver/
│   │   ├── train_silver.parquet                   (regenerable)
│   │   └── test_silver.parquet
│   └── gold/
│       ├── train_gold.parquet                     (regenerable)
│       └── test_gold.parquet
├── notebooks/
│   └── 01_eda.ipynb                               análisis exploratorio
├── src/
│   ├── data_prep.py        raw   -> silver         (limpieza determinística)
│   ├── features.py         silver -> gold          (features de riesgo)
│   ├── validation.py       split temporal + baseline + demo k-fold vs OOT
│   ├── comparar_leakage.py análisis de la variable con leakage
│   ├── train.py            comparación de modelos (champion-challenger)
│   ├── calibracion.py      calibración del modelo campeón
│   ├── politica.py         política de aprobación + backtest de ganancia
│   ├── predict.py          modelo final -> predictions.csv
│   ├── eda_profiling.py    genera reporte HTML con ydata-profiling
│   └── mlops_dashboard.py  genera panel de monitoreo MLOps (HTML)
├── reports/
│   ├── informe_ejecutivo.md       informe para gerencia de Riesgo
│   ├── referencias.md             fuentes de las decisiones
│   ├── diccionario_datos.md       descripción de variables
│   ├── README-Desafio.md          enunciado original del desafío
│   ├── eda_profiling.html         reporte de perfilado automático
│   ├── mlops_dashboard.html       panel de monitoreo MLOps
│   ├── predictions_example.csv    ejemplo de formato de entrega
│   └── figures/
│       ├── calibracion.png
│       ├── default_por_vintage.png
│       └── reconciliacion_ingreso.png
├── predictions.csv         entregable: id_solicitud, prob_default (12.000 filas)
├── AI_USAGE.md
├── requirements.txt
├── requirements-eda.txt    dependencias opcionales (ydata-profiling)
└── README.md
```

## Cómo reproducir

> **Python:** el pipeline principal se probó con **Python 3.12** (soporta 3.11–3.13).
> El análisis opcional con `ydata-profiling` requiere Python < 3.13 (ver `requirements-eda.txt`).

```bash
# 1. Entorno e instalación
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Pipeline principal (regenera predictions.csv desde la data cruda)
python src/data_prep.py    # raw    -> silver
python src/features.py     # silver -> gold
python src/predict.py      # gold   -> predictions.csv (en la raíz)
```

Análisis de apoyo (opcionales, reproducen los hallazgos del informe). El notebook de EDA
requiere `jupyter` (`pip install jupyter`):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb  # EDA
python src/validation.py        # validación out-of-time vs k-fold
python src/comparar_leakage.py  # con vs sin la variable con leakage
python src/train.py             # comparación de modelos + ensemble
python src/calibracion.py       # calibración
python src/politica.py          # política de aprobación + ganancia
```

## Decisiones metodológicas (resumen)

- **Arquitectura medallion** (raw → silver → gold). Las transformaciones con estado
  (imputación, codificación, escalado) van en el pipeline del modelo, ajustadas solo con
  train, para evitar leakage.
- **Corrección de ingreso** mal registrado (error de unidad ×1000), validada por
  reconciliación de la distribución completa; con indicador de corrección.
- **Validación out-of-time** de tres vías (train / validación / holdout), no k-fold
  aleatorio: train y test están separados en el tiempo.
- **Exclusión de una variable con leakage** (`num_contactos_ult_trimestre`): predecía
  implausiblemente bien y su disponibilidad al momento de decidir no es verificable. Se
  reporta el desempeño honesto sin ella.
- **Modelo:** regresión logística (empató a XGBoost/Random Forest/ensemble y gana en
  interpretabilidad), **calibrado** (isotónica) porque la política usa la probabilidad como
  insumo económico.
- **Política de aprobación** por valor esperado por solicitud; el umbral depende del plazo.

Fundamentos y fuentes en `reports/referencias.md`.