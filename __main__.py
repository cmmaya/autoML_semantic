# automl_lib/__main__.py
"""
Main entry point for the automl_lib package.

Allows the package to be executed as a script via `python -m automl_lib`.
"""

from . import cli

# This is the single point of entry that runs the Typer CLI application.
cli.app()