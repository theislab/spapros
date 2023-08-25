#!/usr/bin/env python
"""Command-line interface."""
import logging
import sys

import click
import rich.logging
from pypi_latest import PypiLatest
from rich import print, traceback

import spapros
from spapros.evaluation.evaluation_pipeline import run_evaluation
from spapros.selection.selection import run_selection

log = logging.getLogger()
spapros_pypi_latest = PypiLatest("spapros", spapros.__version__)


def main() -> None:
    traceback.install()

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

    # Is the latest spapros version installed? Upgrade if not!
    if not spapros_pypi_latest.check_latest():
        print("[bold blue]Run [green]spapros upgrade [blue]to get the latest version.")
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
    """Create a selection of probesets for an h5ad file.

    Args:
        data: Path to the h5ad file
        output: Output path
    """
    run_selection(data, output)


@spapros_cli.command()
@click.argument("data", type=click.Path(exists=True))
@click.argument("probeset", type=click.Path(exists=True))
@click.argument("marker-file", type=click.Path(exists=True))
@click.argument("probeset-ids", nargs=-1)
@click.option("--parameters", type=click.Path(exists=True))
@click.option("--output", "-o", default="./results/")
def evaluation(data, probeset, marker_file, probeset_ids, parameters, output) -> None:
    """Create a selection of probesets for an h5ad file.

    Args:
        data: Path to the h5ad dataset file
        probeset: Path to the probeset file
        marker_file: Path to the marker file
        probeset_ids: Several probeset ids
        parameters: Path to a yaml file containing parameters
        output: Output path
    """
    if not probeset_ids:
        probeset_ids = "all"
    else:
        probeset_ids = list(probeset_ids)

    run_evaluation(
        adata_path=data,
        probeset=probeset,
        marker_file=marker_file,
        probeset_ids=probeset_ids,
        result_dir=output,
        parameters_file=parameters,
    )


@spapros_cli.command(short_help="Check for a newer version of ehrapy and upgrade if required.")
def upgrade() -> None:
    """Checks whether the locally installed version of spapros is the latest & upgrades if not."""
    spapros_pypi_latest.check_upgrade()


if __name__ == "__main__":
    main()  # pragma: no cover
