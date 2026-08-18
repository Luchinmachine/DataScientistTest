# Referencias — fundamentos de las decisiones metodológicas

Fuentes que respaldan las decisiones tomadas en el proyecto. Sirven para el informe
ejecutivo y para defender las decisiones en entrevista.

## Data leakage (exclusión de `num_contactos_ult_trimestre`)

- **Kaufman, S., Rosset, S., Perlich, C., Stitelman, O. (2012).** *Leakage in Data Mining:
  Formulation, Detection, and Avoidance.* ACM Transactions on Knowledge Discovery from Data,
  6(4). DOI: 10.1145/2382577.2382579. (Versión previa: KDD 2011, pp. 556–563.)
  → Define el leakage como información sobre el target no disponible legítimamente al
  predecir; advierte que sobreestima el desempeño y que el modelo falla en producción.
  Propone la "separación aprender-predecir" como defensa.

## Validación out-of-time y split temporal

- **Marín, J. (2025).** *Hamiltonian Neural Networks for Robust Out-of-Time Credit Scoring.*
  arXiv:2410.10182. → OOT como evaluación forward-looking con intervalos no solapados;
  refleja condiciones de despliegue.
- **Sheng et al. (2024).** *A Spatio-Temporal Machine Learning Model for Mortgage Credit
  Risk.* arXiv:2410.02846. → Ventana expansible; tuneo de hiperparámetros sobre una
  validación temporal (año más reciente del train), holdout futuro para evaluar.
- **Federal Reserve (2007).** *Report to the Congress on Credit Scoring.* → El hold-out y el
  KS como métodos estándar de validación de scorecards.

## KS y poder discriminante

- **Siddiqi, N. (2006 / 2017).** *Credit Risk Scorecards / Intelligent Credit Scoring.* Wiley.
  → KS y su interpretación; regla de excluir variables sobre-predictivas (escrutinio de
  valores excesivamente altos).
- **Thomas, L., Edelman, D., Crook, J. (2002/2017).** *Credit Scoring and Its Applications.*
  SIAM. → Tratado de referencia de scoring; KS, Gini, AUC.
- Criterios empíricos de KS (~0.3–0.4) discutidos en la literatura de validación de rating.

## Calibración vs. discriminación (dimensiones de validación)

- **Banco Central Europeo (2019).** *Instructions for reporting the validation results of
  internal models.* → Cuatro dimensiones de validación de PD: proceso de rating,
  calibración, poder discriminante y estabilidad. Discriminación y calibración son distintas.

## MLOps: champion-challenger y monitoreo de drift

- **PDx (2025).** *Adaptive Credit Risk Forecasting Model in Digital Lending using MLOps.*
  arXiv:2512.22305. → Framework champion-challenger con validación out-of-time y
  recalibración para resiliencia ante data drift.
- **PSI (Population Stability Index)** — usado en la industria para monitorear drift de
  variables y de score entre poblaciones (train vs. test / producción).

> Nota: las referencias se citan por su aporte a cada decisión. No reemplazan la lectura de
> las fuentes; sirven como fundamento y punto de partida.
