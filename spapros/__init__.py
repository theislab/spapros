"""Top-level package for spapros."""

__author__ = "Lukas Heumos"
__email__ = "lukas.heumos@posteo.net"
__version__ = "0.1.4"

__all__ = ["se", "ev", "pl", "ut"]

from . import evaluation as ev
from . import plotting as pl
from . import selection as se
from . import util as ut
