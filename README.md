# Desafío Data Scientist — Modelo de Probabilidad de Default (Andina Crédito)

Modelo de probabilidad de *default* (mora 90+ días dentro de 12 meses del desembolso)
para apoyar la decisión de aprobación de créditos de consumo, con una política de
aprobación basada en la economía del producto.

## Estructura del repositorio

```
.
├── data/                       # train.csv, test.csv (incluidos con el desafío)
├── notebooks/
│   └── 01_eda.ipynb            # análisis exploratorio con narrativa (ejecutado)
├── src/
│   ├── data_prep.py            # limpieza y features (etapa siguiente)
│   ├── train.py                # entrenamiento + validación (etapa siguiente)
│   └── predict.py              # genera predictions.csv (etapa siguiente)
├── reports/
│   ├── informe_ejecutivo.md    # informe para el gerente de Riesgo (≤ 2 págs)
│   └── figures/                # gráficos citados en el informe
├── predictions.csv             # entregable: id_solicitud, prob_default (12.000 filas)
├── AI_USAGE.md                 # uso de IA y correcciones aplicadas
├── requirements.txt
└── README.md
```

## Cómo reproducir

```bash
# 1. Crear entorno e instalar dependencias
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Reproducir el análisis exploratorio (genera figuras en reports/figures/)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
# o abrir el notebook y ejecutar: Kernel -> Restart & Run All

# 3. (Etapas siguientes) Entrenar y generar predicciones
# python src/train.py
# python src/predict.py     # regenera predictions.csv en la raíz
```

## Decisiones metodológicas (resumen)

- **Validación out-of-time**, no k-fold aleatorio: el train (2024-01 a 2025-02) y el
  test (2025-02 a 2025-06) están separados en el tiempo; validar de forma aleatoria
  filtraría el futuro y daría una estimación optimista.
- **Corrección de ingreso mal registrado** (error de unidad, ×1000) validada por
  reconciliación de la distribución completa; se añade un indicador de corrección.
- **Tendencia de default al alza** (7% → 13% en el train): se explora ponderación por
  recencia para reflejar el régimen de riesgo actual.

*(Este README se completará a medida que se agreguen las etapas de modelado.)*
