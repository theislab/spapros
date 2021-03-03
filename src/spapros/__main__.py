#!/usr/bin/env python
"""Command-line interface."""
import click
import spapros
from rich import print
from rich import traceback


@click.command()
@click.version_option()
def main() -> None:
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
    print('[bold blue]Run [green]spapros --help [blue]for an overview of all commands\n')

    spapros_cli()

@click.group()
@click.version_option(spapros.__version__, message=click.style(f'spapros Version: {spapros.__version__}', fg='blue'))
@click.option('-v', '--verbose', is_flag=True, default=False, help='Enable verbose output (print debug statements).')
@click.option("-l", "--log-file", help="Save a verbose log to a file.")
@click.pass_context
def spapros_cli(ctx, verbose, logfile):
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


if __name__ == "__main__":
    traceback.install()
    main(prog_name="spapros")  # pragma: no cover
