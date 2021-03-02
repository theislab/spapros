#!/usr/bin/env python
"""Command-line interface."""
import click
from rich import traceback
from rich import print

@click.command()
@click.version_option()
def main() -> None:
    print("""[bold blue]
███████ ██████   █████  ██████  ██████   ██████  ███████ 
██      ██   ██ ██   ██ ██   ██ ██   ██ ██    ██ ██      
███████ ██████  ███████ ██████  ██████  ██    ██ ███████ 
     ██ ██      ██   ██ ██      ██   ██ ██    ██      ██ 
███████ ██      ██   ██ ██      ██   ██  ██████  ███████ 
                                                         
""")


if __name__ == "__main__":
    traceback.install()
    main(prog_name="spapros")  # pragma: no cover
