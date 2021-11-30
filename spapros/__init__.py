"""Top-level package for spapros."""

__author__ = "Lukas Heumos"
__email__ = "lukas.heumos@posteo.net"
__version__ = "0.1.0"

from spapros.evaluation.evaluation import ProbesetEvaluator
from spapros.selection.selection_procedure import ProbesetSelector
from spapros.plotting import plot as pl

__all__ = ["ProbesetSelector", "ProbesetEvaluator", "pl"]

