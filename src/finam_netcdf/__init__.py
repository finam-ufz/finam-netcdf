"""
FINAM components NetCDF file I/O.

.. toctree::
   :hidden:

   self

Readers
=======

.. autosummary::
   :toctree: generated
   :caption: Readers

    NetCdfReader
    NetCdfStaticReader

Writers
=======

.. autosummary::
   :toctree: generated
   :caption: Writers

    NetCdfPushWriter
    NetCdfTimedWriter

Tools
=====

.. autosummary::
   :toctree: generated
   :caption: Tools

    Variable
"""
from .reader import NetCdfReader, NetCdfStaticReader
from .tools import Variable
from .writer import NetCdfPushWriter, NetCdfTimedWriter

try:
    from ._version import __version__
except ModuleNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "0.0.0.dev0"

__all__ = [
    "NetCdfStaticReader",
    "NetCdfReader",
    "NetCdfPushWriter",
    "NetCdfTimedWriter",
    "Variable",
]
