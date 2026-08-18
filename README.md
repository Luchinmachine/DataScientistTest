# Desafío Data Scientist — Modelo de Probabilidad de Default (Andina Crédito)

Modelo de probabilidad de *default* (mora 90+ días dentro de 12 meses del desembolso)
para apoyar la decisión de aprobación de créditos de consumo, con una política de
aprobación basada en la economía del producto.

## Estructura del repositorio

```
.
├── data/
│   ├── raw/                    # data cruda del warehouse (versionada): train.csv, test.csv
│   ├── silver/                 # data limpia y tipada (regenerable, parquet)
│   └── gold/                   # capa de negocio / features para el modelo (regenerable)
├── notebooks/
│   └── 01_eda.ipynb            # análisis exploratorio con narrativa (ejecutado)
├── src/
│   ├── data_prep.py            # raw -> silver: limpieza determinística
│   ├── features.py             # silver -> gold: features de negocio (etapa siguiente)
│   ├── train.py                # entrenamiento + validación out-of-time (etapa siguiente)
│   └── predict.py              # genera predictions.csv (etapa siguiente)
├── reports/
│   ├── informe_ejecutivo.md    # informe para el gerente de Riesgo (<= 2 págs)
│   ├── readme-desafio          # enunciado original del desafío
│   └── figures/                # gráficos citados en el informe
├── predictions.csv             # entregable: id_solicitud, prob_default (12.000 filas)
├── AI_USAGE.md                 # uso de IA y correcciones aplicadas
├── requirements.txt
└── README.md
```

## Arquitectura de datos (medallion)

- **raw** → data tal como llega del warehouse. Inmutable, versionada.
- **silver** → data limpia, validada y tipada (`src/data_prep.py`).
- **gold** → capa de negocio con features, lista para el modelo (`src/features.py`).

Las capas silver y gold son artefactos **regenerables** desde raw + código; por eso no
se versionan. Las transformaciones *con estado* (imputación, codificación, escalado) NO
viven en las capas de datos: van en el pipeline del modelo, ajustadas solo con train,
para evitar leakage.

## Cómo reproducir

```bash
# 1. Entorno e instalación
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Generar la capa silver desde raw
python src/data_prep.py

# 3. Análisis exploratorio (genera figuras en reports/figures/)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

# 4. (Etapas siguientes) features, entrenamiento y predicción
# python src/features.py
# python src/train.py
# python src/predict.py     # regenera predictions.csv en la raíz
```

## Decisiones metodológicas (resumen)

- **Validación out-of-time**, no k-fold aleatorio: train (2024-01 a 2025-02) y test
  (2025-02 a 2025-06) están separados en el tiempo; validar aleatorio filtraría el futuro.
- **Corrección de ingreso mal registrado** (error de unidad, ×1000) validada por
  reconciliación de la distribución completa; se añade un indicador de corrección.
- **Tendencia de default al alza** (7% → 13% en el train): se explora ponderación por
  recencia para reflejar el régimen de riesgo actual.

*(Este README se completará a medida que se agreguen las etapas de modelado.)*