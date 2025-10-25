import logging
import time
import operator
import numpy as np
import pandas as pd
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

# ===================================================
# Internal Imports
# ===================================================
from orchestration.multi_script_optimizer import MultiScriptOptimizer, OptimizerConfig
from knowledge import populate_graph
from orchestration.retriever_orchestrator import run_semantic_retriever
from mutation.strategies.hyperparameters import RandomHyperparameterMutation
from mutation.strategies.structure import StructuralSwapStrategy
from mutation.strategies.gp_feature_engineering import GPFeatureEngineeringStrategy
from utils.code_normalizer import normalize_scripts

# ===================================================
# Typer CLI setup
# ===================================================
app = typer.Typer(
    name="automl-optimizer",
    help="An AutoML system for evolving and ensembling ML scripts.",
    add_completion=False
)
console = Console()


# ===================================================
# Enums
# ===================================================
class OptimizationGoal(str, Enum):
    maximize = "maximize"
    minimize = "minimize"


# ===================================================
# Utility: Build mutation strategies dynamically
# ===================================================
def build_strategies(data_path: Path, use_gp: bool = True):
    """Return a list of active mutation strategies based on the dataset."""
    strategies = []

    # Always include core strategies
    hparam_strategy = RandomHyperparameterMutation()
    structural_strategy = StructuralSwapStrategy()
    strategies.extend([hparam_strategy, structural_strategy])

    # Conditional GP feature engineering
    if use_gp:
        try:
            df = pd.read_csv(data_path)
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                def protected_log(x):
                    return np.log(np.abs(x) + 1e-6)

                GP_FUNCTIONS = [
                    ("add", operator.add, 2),
                    ("log", protected_log, 1)
                ]
                GP_TERMINALS = [
                    (col, (lambda c: lambda df: df[c].fillna(0).values)(col))
                    for col in numeric_cols
                ]

                gp_strategy = GPFeatureEngineeringStrategy(GP_FUNCTIONS, GP_TERMINALS)
                strategies.append(gp_strategy)
                console.print(f"[green]✅ GPFeatureEngineeringStrategy enabled ({len(numeric_cols)} numeric cols).[/green]")
            else:
                console.print("[yellow]⚠ No numeric columns detected, skipping GP strategy.[/yellow]")
        except Exception as e:
            console.print(f"[red]⚠ Failed to build GP strategy: {e}[/red]")

    return strategies


# ===================================================
# MAIN COMMAND
# ===================================================
@app.command()
def run(
    scripts: List[Path] = typer.Argument(..., help="Paths to base scripts.", exists=True),
    data_path: Path = typer.Option(..., "--data", "-d", help="Dataset path.", exists=True),
    metric_to_optimize: str = typer.Option(..., "--metric", "-m", help="Metric to optimize."),
    optimization_goal: OptimizationGoal = typer.Option("maximize", "--goal", help="Optimization goal."),
    generations: int = typer.Option(5, "--generations", "-g"),
    population: int = typer.Option(4, "--population", "-p"),
    knowledge_graph_path: Optional[Path] = typer.Option("knowledge_graph.pkl", "--kg-path", help="Path to KG.", exists=True),
    retriever_enabled: bool = typer.Option(False, "--retriever/--no-retriever", help="Enable GPT-5 Retriever step."),
    use_gp: bool = typer.Option(True, "--gp/--no-gp", help="Enable GP-based feature engineering.")
):
    """Run the AutoML optimizer with optional GPT-5 Retriever pre-step."""

    start_time = time.time()
    console.rule("[bold green]AutoML Optimization Run[/bold green]")

    # ===================================================
    # ① Optional Retriever Integration
    # ===================================================
    if retriever_enabled:
        console.rule("[bold magenta]Running Semantic Retriever (GPT-5 + web_search)[/bold magenta]")
        try:
            best_initial_script = run_semantic_retriever(data_path, scripts[0], metric_to_optimize)
            if best_initial_script:
                scripts = [best_initial_script]
                console.print(f"[green]✅ Retriever produced optimized initial script.[/green]")
        except Exception as e:
            console.print(f"[red]Retriever failed: {e}[/red]")

    # ===================================================
    # ② Mutation Strategies
    # ===================================================
    all_strategies = build_strategies(data_path, use_gp=use_gp)
    # ===================================================
    # 🧩 Normalize scripts before optimizer setup
    # ===================================================
    scripts = normalize_scripts(scripts)
    # ===================================================
    # ③ Optimizer Setup
    # ===================================================
    config = OptimizerConfig(num_generations=generations, population_size=population)
    optimizer = MultiScriptOptimizer(
        script_paths=scripts,
        data_path=data_path,
        config=config,
        metric_to_optimize=metric_to_optimize,
        optimization_goal=optimization_goal.value,
        strategies=all_strategies,
        knowledge_graph_path=knowledge_graph_path,
    )

    # ===================================================
    # ④ Run Optimization
    # ===================================================
    with console.status("[bold yellow]Evolving scripts...[/bold yellow]", spinner="dots"):
        final_output = optimizer.run()

    # ===================================================
    # ⑤ Display Results
    # ===================================================
    duration = time.time() - start_time
    console.rule(f"[bold green]Optimization Finished in {duration:.2f}s[/bold green]")

    if final_output and final_output.get("best_single_model"):
        console.print(Panel.fit("[bold blue]Best Model[/bold blue]"))
        console.print(Pretty(final_output["best_single_model"]))
    else:
        console.print("[bold red]No successful models found.[/bold red]")


# ===================================================
# BUILD KNOWLEDGE GRAPH
# ===================================================
@app.command()
def build_kg(output_path: Path = typer.Option("knowledge_graph.pkl", "--output", "-o", help="Save KG here.")):
    """Populate the Knowledge Graph."""
    console.rule("[bold green]Building Knowledge Graph[/bold green]")
    with console.status("[bold yellow]Parsing ML libraries...[/bold yellow]", spinner="dots"):
        populate_graph.populate(output_path)
    console.print(f"[bold green]✅ Knowledge Graph saved to '{output_path}'[/bold green]")


# ===================================================
# ENTRYPOINT
# ===================================================
if __name__ == "__main__":
    app()
