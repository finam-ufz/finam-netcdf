from datetime import datetime

import finam as fm
from finam_plot import ContourPlot

from finam_netcdf import NetCdfReader


def to_time_step(tick, _last_time, _last_index):
    year = start.year + tick // 12
    month = 1 + tick % 12
    return datetime(year, month, 1), tick % 12


start = datetime(2000, 1, 1)
path = "tests/data/lai.nc"

reader = NetCdfReader(path, ["lai"], time_callback=to_time_step)

viewer = ContourPlot()

composition = fm.Composition([reader, viewer])
composition.initialize()

reader.outputs["lai"] >> fm.adapters.LinearTime() >> viewer.inputs["Grid"]

composition.run(end_time=datetime(2005, 1, 1))
