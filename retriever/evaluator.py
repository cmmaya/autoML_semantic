# ===============================================================
# evaluator.py
# Evalúa localmente los scripts generados por GPT-5
# ===============================================================

import subprocess
import re, json
import time
from pathlib import Path
from typing import List
from rich.console import Console
from utils.auto_debugger import AutoDebugger

console = Console()



class CandidateEvaluator:
    def __init__(self, gpt_model="gpt-5", debug_loops=1):
        self.cache_dir = Path("retriever/cache")
        self.log_path = self.cache_dir / "results_log.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.debugger = AutoDebugger(gpt_model=gpt_model, max_loops=debug_loops)

    # ===============================================================
    def _run_script(self, script_path: Path, is_baseline=False, debugged=False) -> float:
        """
        Executes a candidate script, extracts its AUC,
        and triggers the debugger if it fails or returns 0.0.
        """
        label = "baseline" if is_baseline else "candidate"

        try:
            out = subprocess.check_output(
                ["python", str(script_path)],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=180
            )
            out = out.strip()

            # 1️⃣ Try parsing JSON { "auc": ... } or {'auc': ...}
            try:
                json_str = out[out.find("{"):out.find("}")+1]
                if json_str:
                    try:
                        metric_json = json.loads(json_str)
                    except json.JSONDecodeError:
                        fixed = json_str.replace("'", '"')
                        metric_json = json.loads(fixed)
                    score = float(metric_json.get("auc", 0.0))
                    # 👇 If the score is 0.0, treat as invalid and debug
                    if score == 0.0 and not debugged:
                        console.print(f"[red]⚠ {script_path.name} returned 0 AUC. Triggering debugger...[/red]")
                        repaired = self.debugger.debug_script(script_path, out)
                        console.print(f"[cyan]🔁 Re-running repaired script...[/cyan]")
                        return self._run_script(repaired, is_baseline, debugged=True)
                    return score

            except json.JSONDecodeError:
                pass

            # 2️⃣ Regex fallback: "auc: 0.6648"
            match = re.search(r"auc[:=]\s*([0-9.]+)", out)
            if match:
                score = float(match.group(1))
                if score == 0.0 and not debugged:
                    console.print(f"[red]⚠ {script_path.name} returned 0 AUC. Triggering debugger...[/red]")
                    repaired = self.debugger.debug_script(script_path, out)
                    console.print(f"[cyan]🔁 Re-running repaired script...[/cyan]")
                    return self._run_script(repaired, is_baseline, debugged=True)
                return score

            # 3️⃣ No valid metric found
            console.print(f"[yellow]⚠ No metric found in {script_path.name} output.[/yellow]")
            if not debugged:
                repaired = self.debugger.debug_script(script_path, out)
                console.print(f"[cyan]🔁 Re-running repaired script...[/cyan]")
                return self._run_script(repaired, is_baseline, debugged=True)
            return 0.0

        # ===============================================================
        except subprocess.CalledProcessError as e:
            console.print(f"[red]❌ {script_path.name} failed to run. Triggering debugger...[/red]")
            console.print(f"[dim]{e.output[:300]}...[/dim]")
            if not debugged:
                repaired = self.debugger.debug_script(script_path, e.output)
                console.print(f"[cyan]🔁 Re-running repaired script...[/cyan]")
                return self._run_script(repaired, is_baseline, debugged=True)
            return 0.0

        except subprocess.TimeoutExpired:
            console.print(f"[red]⏱ {script_path.name} exceeded maximum time.[/red]")
            return 0.0

        except Exception as ex:
            console.print(f"[red]⚠ Unexpected error executing {script_path.name}: {ex}[/red]")
            return 0.0


    def evaluate(self, candidates: List[Path], data_path: str, metric: str) -> Path:
        """Evalúa todos los scripts candidatos y devuelve el mejor."""
        evaluations = []

        for script in candidates:
            console.print(f"[yellow]▶ Ejecutando {script.name}...[/yellow]")
            start = time.time()
            score = self._run_script(script)
            duration = time.time() - start

            evaluations.append({
                "script": script.name,
                "score": score,
                "duration": duration
            })

            if score > 0:
                console.print(f"[green]✅ {script.name}: {metric} = {score:.4f}[/green]")
            else:
                console.print(f"[red]⚠ {script.name}: no produjo resultados válidos.[/red]")

        if not evaluations:
            console.print("[red]❌ Ningún candidato fue evaluado correctamente.[/red]")
            return None

        # Ordena por métrica descendente
        best = max(evaluations, key=lambda x: x["score"])
        log = {
            "metric": metric,
            "evaluations": evaluations,
            "best_result": best
        }

        # Guarda histórico de resultados
        with open(self.log_path, "w") as f:
            json.dump(log, f, indent=2)

        console.print(f"[bold green]🏆 Mejor script: {best['script']} ({best['score']:.4f})[/bold green]")
        return self.cache_dir / best["script"]
