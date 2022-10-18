from datetime import datetime, timedelta

import finam as fm
from finam.adapters.time import LinearInterpolation
from finam_plot import ContourPlot

from finam_netcdf import Layer
from finam_netcdf.reader import NetCdfTimeReader

if __name__ == "__main__":
    path = "tests/data/lai.nc"

    reader = NetCdfTimeReader(
        path, {"LAI": Layer(var="lai", xyz=("lon", "lat"))}, time_var="time"
    )

    viewer = ContourPlot()

    composition = fm.Composition([reader, viewer])
    composition.initialize()

    _ = reader.outputs["LAI"] >> LinearInterpolation() >> viewer.inputs["Grid"]

    composition.run(datetime(1901, 1, 1, 0, 12))
