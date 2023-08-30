from datetime import datetime

import finam as fm
import matplotlib.pyplot as plt
from finam_plot import ContourPlot

from finam_netcdf import Layer, NetCdfStaticReader

if __name__ == "__main__":
    path = "tests/data/lai.nc"

    reader = NetCdfStaticReader(
        path,
        {"LAI": Layer(var="lai", xyz=("lon", "lat"), fixed={"time": 0})},
    )

    viewer = ContourPlot()

    composition = fm.Composition([reader, viewer])
    composition.initialize()

    reader.outputs["LAI"] >> viewer.inputs["Grid"]

    composition.run()

    plt.show(block=True)
