#!/usr/bin/env python
"""Command-line interface."""
import click
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


if __name__ == "__main__":
    traceback.install()
    main(prog_name="spapros")  # pragma: no cover
