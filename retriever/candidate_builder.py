# ===============================================================
# candidate_builder.py
# Limpia y adapta los scripts generados por el RetrieverAgent
# ===============================================================

from pathlib import Path
import re
from typing import List, Dict
from rich.console import Console

console = Console()


class CandidateBuilder:
    def __init__(self):
        self.cache_dir = Path("retriever/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_code(self, code: str, data_path: str) -> str:
        """Normaliza los scripts generados desde JSON ya limpio y corrige errores comunes de formato."""
        if not code:
            return ""

        # ----------------------------------------------------------
        # 1️ Quitar delimitadores tipo ```python ... ``` o comillas
        # ----------------------------------------------------------
        code = re.sub(r"^```(?:python)?|```$", "", code.strip(), flags=re.MULTILINE)
        code = code.strip().strip('"').strip("'")

        # ----------------------------------------------------------
        # 2️ Reemplazar tabulaciones REALES (\t) por 4 espacios
        # ----------------------------------------------------------
        code = code.replace("\t", "    ")  # tab literal → espacios

        # ---------------------------------------------------------
        # 3️ Restaurar saltos de línea y limpiar escapes residuales
        # ----------------------------------------------------------
        code = code.replace("\\n", "\n").replace("\\r", "")
        code = code.replace("\\t", "    ")

        # ----------------------------------------------------------
        # 4️ Corregir concatenaciones accidentales
        # ----------------------------------------------------------
        code = re.sub(r"\)\\?y", ")\ny", code)
        code = re.sub(r"(?<=\))\s*y\s*=", "\ny =", code)

        # ----------------------------------------------------------
        # 5️ Sustituir pd.read_csv(...) por la ruta real
        # ----------------------------------------------------------
        code = re.sub(
            r"pd\.read_csv\([^)]*\)",
            f"pd.read_csv(r'{data_path}')",
            code
        )

        # 🔸 EXTRA: eliminar tabs ocultos o espacios no imprimibles en rutas
        code = re.sub(r"data\s+rain\.csv", "data/train.csv", code)
        code = re.sub(r"data\s+train\.csv", "data/train.csv", code)
        code = code.replace("data\t", "data/")  # tab real → slash

        # ----------------------------------------------------------
        # 6️ Normalización final
        # ----------------------------------------------------------
        code = re.sub(r"\r\n|\r", "\n", code).strip() + "\n"

        # ----------------------------------------------------------
        # 7️ Insertar print si no hay
        # ----------------------------------------------------------
        if not re.search(r"print\s*\(", code):
            code += "\nprint({'auc': model.score(X_test, y_test)})\n"

        return code


    def build_candidates(self, examples: List[Dict], data_path: str) -> List[Path]:
        """Genera archivos .py listos para evaluar localmente."""
        candidates = []
        for i, ex in enumerate(examples, start=1):
            name = re.sub(r'[^a-zA-Z0-9_]', '_', ex.get("name", f"candidate_{i}"))
            file_path = self.cache_dir / f"{name}.py"
            code = self._sanitize_code(ex.get("code_snippet", ""), data_path)
            with open(file_path, "w") as f:
                f.write(code)
            candidates.append(file_path)
            console.print(f"[green]💾 Guardado {file_path}[/green]")
        return candidates
