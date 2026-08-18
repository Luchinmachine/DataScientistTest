"""
EDA automatizado (HTML) con ydata-profiling.

COMPLEMENTO del análisis manual en notebooks/01_eda.ipynb, no reemplazo: el perfilado
automático da un primer barrido de la data cruda (distribuciones, faltantes,
correlaciones), pero el criterio —diagnosticar el error de unidad del ingreso, detectar
el leakage, chequear la madurez del target— vive en el análisis manual.

OPCIONAL: no es parte del pipeline principal. Requiere requirements-eda.txt (Python < 3.13).

Uso:
    pip install -r requirements-eda.txt
    python src/eda_profiling.py
"""

from pathlib import Path
import pandas as pd
from ydata_profiling import ProfileReport

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
REPORTS = Path(__file__).resolve().parent.parent / "reports"


def main():
    df = pd.read_csv(RAW / "train.csv")
    profile = ProfileReport(
        df,
        title="EDA automatizado — Andina Crédito (train)",
        # Sin matriz de scatter (interactions): infla el HTML y aporta poco aquí.
        interactions={"continuous": False},
        progress_bar=False,
        dataset={"description": "Perfilado de la data cruda de entrenamiento. "
                                "Complemento del análisis manual (notebooks/01_eda.ipynb)."},
    )
    salida = REPORTS / "eda_profiling.html"
    profile.to_file(salida)
    print(f"Reporte generado -> {salida}")


if __name__ == "__main__":
    main()
