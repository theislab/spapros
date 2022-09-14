#!/usr/bin/env python
"""Command-line interface."""
import click
from rich import traceback


@click.command()
@click.version_option(version="0.1.1", message=click.style("spapros Version: 0.1.1"))
def main() -> None:
    """spapros."""


if __name__ == "__main__":
    traceback.install()
    main(prog_name="spapros")  # pragma: no cover
