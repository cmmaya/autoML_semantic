# ===============================================================
# orchestration/retriever_orchestrator.py
# ===============================================================

from retriever.candidate_builder import CandidateBuilder
from retriever.retriever_agent import RetrieverAgent
from retriever.evaluator import CandidateEvaluator
from utils.dataset_summary import build_dataset_summary
from rich.console import Console

console = Console()

def run_semantic_retriever(data_path, base_script_path, metric):
    """Ejecuta el flujo completo del retriever: LLM → builder → evaluator → selección final."""
    dataset_summary = build_dataset_summary(data_path)

    agent = RetrieverAgent()
    builder = CandidateBuilder()
    evaluator = CandidateEvaluator()

    # === Paso 1: generar candidatos ===
    examples = agent.retrieve_contextual_examples(dataset_summary)
    if not examples:
        console.print("[red]❌ No se pudieron generar ejemplos válidos.[/red]")
        return base_script_path

    candidates = builder.build_candidates(examples, data_path)

    # === Paso 2: evaluar candidatos ===
    console.rule("[bold cyan]Evaluando candidatos generados...[/bold cyan]")
    best_candidate = evaluator.evaluate(candidates, data_path, metric)

    # === Paso 3: evaluar baseline original ===
    console.rule("[bold cyan]Evaluando script original (baseline)...[/bold cyan]")
    baseline_score = evaluator._run_script(base_script_path, is_baseline=True)
    console.print(f"[blue]📊 Baseline {base_script_path.name}: {metric} = {baseline_score:.4f}[/blue]")

    # === Paso 4: comparar y decidir ===
    if best_candidate is None:
        console.print("[red]❌ No hubo candidatos válidos, se mantiene el script original.[/red]")
        return base_script_path

    # Recuperar el score del mejor candidato desde el log
    import json
    log_path = evaluator.log_path
    best_score = 0.0
    if log_path.exists():
        with open(log_path, "r") as f:
            log_data = json.load(f)
            best_score = log_data.get("best_result", {}).get("score", 0.0)

    console.rule("[bold magenta]Comparando resultados...[/bold magenta]")
    if best_score > baseline_score:
        console.print(f"[green]✅ El candidato superó al baseline ({best_score:.4f} > {baseline_score:.4f})[/green]")
        return best_candidate
    else:
        console.print(f"[yellow]⚖ El baseline es mejor o igual ({baseline_score:.4f} ≥ {best_score:.4f}). Se mantiene el original.[/yellow]")
        return base_script_path
