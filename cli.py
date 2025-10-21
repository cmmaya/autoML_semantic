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

from .orchestration.multi_script_optimizer import MultiScriptOptimizer, OptimizerConfig
from .knowledge import populate_graph

# --- CLI Setup ---
app = typer.Typer(
    name="automl-optimizer",
    help="An S-tier AutoML system for evolving and ensembling ML scripts.",
    add_completion=False
)
console = Console()

class OptimizationGoal(str, Enum):
    """An enumerable for the optimization goal to ensure valid choices."""
    maximize = "maximize"
    minimize = "minimize"


@app.command()
def run(
    scripts: List[Path] = typer.Argument(
        ..., # ... means this argument is required
        help="Paths to the user scripts to be optimized.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    data_path: Path = typer.Option(
        ..., "--data", "-d",
        help="Path to the training dataset (CSV format).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    metric_to_optimize: str = typer.Option(
        ..., "--metric", "-m",
        help="The key in the script's returned metrics dict to optimize."
    ),
    optimization_goal: OptimizationGoal = typer.Option(
        "maximize", "--goal",
        help="Whether to maximize or minimize the target metric.",
        case_sensitive=False,
    ),
    generations: int = typer.Option(5, "--generations", "-g", help="Number of generations to run."),
    population: int = typer.Option(4, "--population", "-p", help="Population size for each generation."),
    knowledge_graph_path: Optional[Path] = typer.Option(
        "knowledge_graph.pkl", "--kg-path",
        help="Path to the pre-populated knowledge graph.",
        exists=True,
        file_okay=True,
    ),
):
    """
    Run the end-to-end multi-script optimization process.
    """
    start_time = time.time()
    console.rule("[bold green]Starting AutoML Optimization Run[/bold green]")
    
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
        best_model = final_output["best_single_model"]
        ensemble = final_output["final_ensemble"]
        
        console.print(Panel.fit("[bold blue]Best Single Model Found[/bold blue]"))
        console.print(Pretty(best_model))

        console.print(Panel.fit("[bold blue]Final Ensemble[/bold blue]"))
        console.print(Pretty(ensemble))
    else:
        console.print("[bold red]No successful models were found during the run.[/bold red]")


@app.command()
def build_kg(
    output_path: Path = typer.Option("knowledge_graph.pkl", "--output", "-o", help="Path to save the generated knowledge graph.")
):
    """
    Populate the Knowledge Graph by parsing ML libraries.
    """
    console.rule("[bold green]Building Knowledge Graph[/bold green]")
    with console.status("[bold yellow]Parsing libraries... this may take a while.[/bold yellow]", spinner="dots"):
        populate_graph.populate(output_path)
    console.print(f"[bold green]✅ Knowledge Graph saved to '{output_path}'[/bold green]")


if __name__ == "__main__":
    app()