# automl_lib/cli.py
"""
The main Command-Line Interface (CLI) for the AutoML system.

This module provides a user-friendly command line entry point for running
the multi-script optimizer, populating the knowledge graph, and more.
"""
import logging
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from orchestration.multi_script_optimizer import MultiScriptOptimizer, OptimizerConfig
from knowledge import populate_graph
# --- NEW ---
from orchestration.retriever_orchestrator import run_semantic_retriever

app = typer.Typer(
    name="automl-optimizer",
    help="An S-tier AutoML system for evolving and ensembling ML scripts.",
    add_completion=False
)
console = Console()

class OptimizationGoal(str, Enum):
    maximize = "maximize"
    minimize = "minimize"

@app.command()
def run(
    scripts: List[Path] = typer.Argument(..., help="Paths to scripts.", exists=True),
    data_path: Path = typer.Option(..., "--data", "-d", help="Dataset path.", exists=True),
    metric_to_optimize: str = typer.Option(..., "--metric", "-m", help="Metric key to optimize."),
    optimization_goal: OptimizationGoal = typer.Option("maximize", "--goal", help="Optimization goal."),
    generations: int = typer.Option(5, "--generations", "-g"),
    population: int = typer.Option(4, "--population", "-p"),
    knowledge_graph_path: Optional[Path] = typer.Option("knowledge_graph.pkl", "--kg-path", help="Path to KG.", exists=True),
    retriever_enabled: bool = typer.Option(True, "--retriever", help="Enable GPT-5 Retriever step.")
):
    """Run the AutoML optimizer with optional GPT-5 Retriever pre-step."""
    start_time = time.time()
    console.rule("[bold green]AutoML Optimization Run[/bold green]")

    # === Optional Retriever Integration ===
    if retriever_enabled:
        console.rule("[bold magenta]Running Semantic Retriever (GPT-5 + web_search)[/bold magenta]")
        try:
            best_initial_script = run_semantic_retriever(data_path, scripts[0], metric_to_optimize)
            if best_initial_script:
                scripts = [best_initial_script]
                console.print(f"[green]✅ Retriever produced optimized initial script.[/green]")
        except Exception as e:
            console.print(f"[red]Retriever failed: {e}[/red]")

    # === Optimizer ===
    config = OptimizerConfig(num_generations=generations, population_size=population)
    optimizer = MultiScriptOptimizer(
        script_paths=scripts,
        data_path=data_path,
        config=config,
        metric_to_optimize=metric_to_optimize,
        optimization_goal=optimization_goal.value,
        knowledge_graph_path=knowledge_graph_path
    )
    with console.status("[bold yellow]Evolving scripts...[/bold yellow]", spinner="dots"):
        final_output = optimizer.run()
    
    duration = time.time() - start_time
    console.rule(f"[bold green]Optimization Finished in {duration:.2f}s[/bold green]")

    if final_output and final_output.get("best_single_model"):
        console.print(Panel.fit("[bold blue]Best Model[/bold blue]"))
        console.print(Pretty(final_output["best_single_model"]))
    else:
        console.print("[bold red]No successful models found.[/bold red]")

@app.command()
def build_kg(output_path: Path = typer.Option("knowledge_graph.pkl", "--output", "-o", help="Save KG here.")):
    """Populate the Knowledge Graph."""
    console.rule("[bold green]Building Knowledge Graph[/bold green]")
    with console.status("[bold yellow]Parsing ML libraries...[/bold yellow]", spinner="dots"):
        populate_graph.populate(output_path)
    console.print(f"[bold green]✅ Knowledge Graph saved to '{output_path}'[/bold green]")

if __name__ == "__main__":
    app()
