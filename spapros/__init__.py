"""Top-level package for spapros."""

__author__ = "Lukas Heumos"
__email__ = "lukas.heumos@posteo.net"
__version__ = "0.1.0"

import spapros.selection.selection_procedure as se
import spapros.evaluation.evaluation as ev
from spapros.plotting import plot as pl

__all__ = ["se", "ev", "pl"]
