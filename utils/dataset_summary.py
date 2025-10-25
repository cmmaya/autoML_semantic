# ================================================================
# utils/dataset_summary.py
# Construye un resumen estructurado del dataset para el retriever
# ================================================================

import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from typing import Dict
from rich.console import Console

console = Console()


def build_dataset_summary(data_path: str, metric: str = "auc") -> Dict:
    """
    Analiza localmente el dataset y devuelve un resumen estructurado
    que será usado por el RetrieverAgent para adaptar los scripts.

    Parámetros:
        data_path (str): Ruta del CSV.
        metric (str): Métrica objetivo (por defecto 'auc').

    Retorna:
        dict: Estructura con metadatos del dataset.
    """

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        console.print(f"[red]Error al leer el dataset: {e}[/red]")
        return {}

    # ----------------------------------------------------------
    # ① Información general
    # ----------------------------------------------------------
    n_rows, n_cols = df.shape
    console.print(f"[yellow]📊 Dataset cargado: {n_rows} filas × {n_cols} columnas[/yellow]")

    # ----------------------------------------------------------
    # ② Detección de tipos de columnas
    # ----------------------------------------------------------
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # ----------------------------------------------------------
    # ③ Detección de columna target (heurística simple)
    # ----------------------------------------------------------
    target_col = None
    for candidate in ["target", "label", "class", "y", "output", "Survived"]:
        if candidate in df.columns:
            target_col = candidate
            break

    if not target_col:
        # Si no se encuentra, asumimos la última columna como target
        target_col = df.columns[-1]

    # ----------------------------------------------------------
    # ④ Inferencia del tipo de tarea
    # ----------------------------------------------------------
    unique_vals = df[target_col].nunique()
    if unique_vals <= 20 and df[target_col].dtype in [np.int64, np.int32, np.object_]:
        task_type = "classification"
    else:
        task_type = "regression"

    # ----------------------------------------------------------
    # ⑤ Hash único del dataset (para nombrar caché)
    # ----------------------------------------------------------
    dataset_id = hashlib.sha1(Path(data_path).name.encode()).hexdigest()[:8]

    # ----------------------------------------------------------
    # ⑥ Construcción del resumen estructurado
    # ----------------------------------------------------------
    summary = {
        "path": str(data_path),
        "dataset_id": dataset_id,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "task_type": task_type,
        "metric": metric,
        "columns": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "target": target_col
        },
        "target_unique_values": unique_vals,
    }

    # ----------------------------------------------------------
    # ⑦ Diagnóstico en consola
    # ----------------------------------------------------------
    console.print("[bold cyan]✅ Resumen del dataset:[/bold cyan]")
    console.print_json(data=summary)

    return summary
