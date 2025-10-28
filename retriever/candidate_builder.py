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
        """Reemplaza la ruta del dataset en los scripts generados."""
        # Reemplaza cualquier pd.read_csv('...') con la ruta real
        code = re.sub(r"pd\.read_csv\([^)]*\)", f"pd.read_csv(r'{data_path}')", code)
        # Asegura que imprima un JSON con métricas
        if "print(" not in code:
            code += "\n\nprint({'val_accuracy': model.score(X_test, y_test)})"
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
