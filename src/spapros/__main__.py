#!/usr/bin/env python
"""Command-line interface."""
import logging
import sys

import click
import rich.logging
from rich import print
from rich import traceback

import spapros
from spapros.evaluation.evaluation_pipeline import run_evaluation
from spapros.selection.selection import run_selection

log = logging.getLogger()


def main() -> None:
    traceback.install()
    # COOKIETEMPLE TODO: Remove the warnings filter!
    import warnings

    warnings.filterwarnings("ignore")

    """Main entry point for spapros."""
    print(
        """[bold blue]
███████ ██████   █████  ██████  ██████   ██████  ███████ 
██      ██   ██ ██   ██ ██   ██ ██   ██ ██    ██ ██      
███████ ██████  ███████ ██████  ██████  ██    ██ ███████ 
     ██ ██      ██   ██ ██      ██   ██ ██    ██      ██ 
███████ ██      ██   ██ ██      ██   ██  ██████  ███████ 
                                                         
"""
    )
    print("[bold blue]Run [green]spapros --help [blue]for an overview of all commands\n")

    spapros_cli(prog_name="spapros")


@click.group()
@click.version_option(
    spapros.__version__,
    message=click.style(f"spapros Version: {spapros.__version__}", fg="blue"),
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose output (print debug statements).",
)
@click.option("-l", "--log-file", help="Save a verbose log to a file.")
@click.pass_context
def spapros_cli(ctx, verbose, log_file):
    """
    Create state of the art probesets for spatial transcriptomics projects.
    """
    # Set the base logger to output DEBUG
    log.setLevel(logging.DEBUG)

    # Set up logs to the console
    log.addHandler(
        rich.logging.RichHandler(
            level=logging.DEBUG if verbose else logging.INFO,
            console=rich.console.Console(file=sys.stderr),
            show_time=True,
            markup=True,
        )
    )

    # Set up logs to a file if we asked for one
    if log_file:
        log_fh = logging.FileHandler(log_file, encoding="utf-8")
        log_fh.setLevel(logging.DEBUG)
        log_fh.setFormatter(logging.Formatter("[%(asctime)s] %(name)-20s [%(levelname)-7s]  %(message)s"))
        log.addHandler(log_fh)


@spapros_cli.command()
@click.argument("data", type=click.Path(exists=True))
@click.option("--output", "-o", default="./results/")
def selection(data, output) -> None:
    """
    Create a selection of probesets for an h5ad file
    Args:
        data: Path to the h5ad file
        output: Output path
    """
    run_selection(data, output)


@spapros_cli.command()
@click.argument("probeset", type=click.Path(exists=True))
@click.option("--output", "-o", default="./results/")
def evaluation(probeset, output) -> None:
    """
    Create a selection of probesets for an h5ad file
    Args:
        probeset: Path to the probeset file
        output: Output path
    """
    run_evaluation(probeset, output)


if __name__ == "__main__":
    main()  # pragma: no cover
