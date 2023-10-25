from datetime import datetime

import finam as fm
from finam_plot import ContourPlot

from finam_netcdf import NetCdfReader

path = "tests/data/lai.nc"

reader = NetCdfReader(path, outputs=["lai"])

viewer = ContourPlot()

composition = fm.Composition([reader, viewer])
composition.initialize()

reader.outputs["lai"] >> fm.adapters.LinearTime() >> viewer.inputs["Grid"]

composition.run(end_time=datetime(1901, 1, 1, 0, 12))
