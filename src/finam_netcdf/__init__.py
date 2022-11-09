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

    Layer
"""
from .reader import NetCdfReader, NetCdfStaticReader
from .tools import Layer
from .writer import NetCdfPushWriter, NetCdfTimedWriter

__all__ = [
    "NetCdfStaticReader",
    "NetCdfReader",
    "NetCdfPushWriter",
    "NetCdfTimedWriter",
    "Layer",
]
