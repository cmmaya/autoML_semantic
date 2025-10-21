import logging
import time
import operator
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.syntax import Syntax

from orchestration.multi_script_optimizer import MultiScriptOptimizer, OptimizerConfig
from knowledge import populate_graph
from mutation.strategies.hyperparameters import RandomHyperparameterMutation
from mutation.strategies.structure import StructuralSwapStrategy
from mutation.strategies.gp_feature_engineering import GPFeatureEngineeringStrategy

# --- CLI Setup ---
app = typer.Typer(
    name="automl-optimizer",
    help="An S-tier AutoML system for evolving and ensembling ML scripts.",
    add_completion=False,
    rich_markup_mode="markdown"
)
console = Console()

class OptimizationGoal(str, Enum):
    maximize = "maximize"
    minimize = "minimize"

@app.command()
def run(
    scripts: List[Path] = typer.Argument(..., help="Paths to the user scripts to be optimized.", exists=True, file_okay=True, readable=True),
    data_path: Path = typer.Option(..., "--data", "-d", help="Path to the training dataset (CSV format).", exists=True, file_okay=True, readable=True),
    metric_to_optimize: str = typer.Option(..., "--metric", "-m", help="The key in the script's returned metrics dict to optimize."),
    optimization_goal: OptimizationGoal = typer.Option("maximize", "--goal", help="Whether to maximize or minimize the target metric.", case_sensitive=False),
    generations: int = typer.Option(5, "--generations", "-g", help="Number of generations to run."),
    population: int = typer.Option(4, "--population", "-p", help="Population size for each generation."),
    knowledge_graph_path: Optional[Path] = typer.Option("knowledge_graph.pkl", "--kg-path", help="Path to the pre-populated knowledge graph.", exists=True, readable=True),
):
    """
    Run the end-to-end multi-script optimization process.
    """
    console.rule("[bold green]Initializing Strategies[/bold green]")

    # --- FIX: Dynamic GP Terminal Generation ---
    try:
        df = pd.read_csv(data_path)
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        # A more robust version might exclude the target column if present
        GP_TERMINALS = [
            (col, (lambda c: lambda df: df[c].fillna(0).values)(col)) for col in numerical_cols
        ]
        console.print(f"✅ Dynamically generated {len(GP_TERMINALS)} GP terminals from numerical columns.")
    except Exception as e:
        console.print(f"[bold red]Error reading data for GP terminal generation: {e}[/bold red]")
        GP_TERMINALS = []
    # --- END FIX ---
    
    def protected_log(x): return np.log(np.abs(x) + 1e-6)
    GP_FUNCTIONS = [('add', operator.add, 2), ('log', protected_log, 1)]

    hparam_strategy = RandomHyperparameterMutation()
    structural_strategy = StructuralSwapStrategy()
    gp_fe_strategy = GPFeatureEngineeringStrategy(GP_FUNCTIONS, GP_TERMINALS)
    all_strategies = [hparam_strategy, structural_strategy]
    if GP_TERMINALS: # Only add the GP strategy if it has columns to work with
        all_strategies.append(gp_fe_strategy)
    
    console.print(Panel.fit(f"[bold yellow]Optimizer armed with {len(all_strategies)} mutation strategies.[/bold yellow]"))

    # --- Run the Optimizer ---
    start_time = time.time()
    console.rule("[bold green]Starting AutoML Optimization Run[/bold green]")
    
    config = OptimizerConfig(num_generations=generations, population_size=population)
    
    optimizer = MultiScriptOptimizer(
        script_paths=scripts,
        data_path=data_path,
        config=config,
        metric_to_optimize=metric_to_optimize,
        optimization_goal=optimization_goal.value,
        strategies=all_strategies,
        knowledge_graph_path=knowledge_graph_path
    )
    
    with console.status("[bold yellow]Evolving scripts...[/bold yellow]", spinner="dots"):
        final_output = optimizer.run()
    
    duration = time.time() - start_time
    console.rule(f"[bold green]Optimization Finished in {duration:.2f}s[/bold green]")

    if final_output and final_output.get("best_single_model_result"):
        console.print(Panel.fit("[bold blue]Best Single Model Found[/bold blue]"))
        console.print(Pretty(final_output["best_single_model_result"]))
        console.print(Panel.fit("[bold blue]Final Ensemble[/bold blue]"))
        console.print(Pretty(final_output["final_ensemble_hparams"]))

        best_code = final_output.get("best_single_model_code")
        if best_code:
            output_code_path = "best_script.py"
            with open(output_code_path, "w") as f:
                f.write(best_code)
            
            console.print(Panel.fit("[bold green]🏆 Best Evolved Code 🏆[/bold green]"))
            console.print(f"The source code for the best model has been saved to: [bold]{output_code_path}[/bold]")
            console.print(Syntax(best_code, "python", theme="monokai", line_numbers=True))
    else:
        console.print("[bold red]No successful models were found during the run.[/bold red]")

@app.command()
def build_kg(
    output_path: Path = typer.Option("knowledge_graph.pkl", "--output", "-o", help="Path to save the generated KG.")
):
    """Populate the Knowledge Graph by parsing ML libraries."""
    console.rule("[bold green]Building Knowledge Graph[/bold green]")
    with console.status("[bold yellow]Parsing libraries... this may take a while.[/bold yellow]", spinner="dots"):
        populate_graph.populate(output_path)
    console.print(f"[bold green]✅ Knowledge Graph saved to '{output_path}'[/bold green]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(message)s')
    app()