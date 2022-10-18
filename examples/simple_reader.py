from datetime import datetime

import finam as fm
import matplotlib.pyplot as plt
from finam_plot import ContourPlot

from finam_netcdf import Layer
from finam_netcdf.reader import NetCdfInitReader

if __name__ == "__main__":
    path = "tests/data/lai.nc"

    reader = NetCdfInitReader(
        path,
        {"LAI": Layer(var="lai", xyz=("lon", "lat"), fixed={"time": 0})},
    )

    viewer = ContourPlot()

    composition = fm.Composition([reader, viewer])
    composition.initialize()

    _ = reader.outputs["LAI"] >> viewer.inputs["Grid"]

    composition.run(datetime(2000, 7, 1))

    plt.show(block=True)
