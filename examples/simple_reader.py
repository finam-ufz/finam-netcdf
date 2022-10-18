from datetime import datetime, timedelta

import finam as fm
from finam_plot import C

from finam_netcdf import Layer
from finam_netcdf.reader import NetCdfInitReader

if __name__ == "__main__":
    path = "tests/data/lai.nc"

    reader = NetCdfInitReader(
        path, {"LAI": Layer(var="lai", xyz=("lon", "lat"), fixed={"time": 0})}
    )

    viewer = grid.TimedGridView(start=datetime(2000, 1, 1), step=timedelta(days=1))

    composition = Composition([reader, viewer])
    composition.initialize()

    _ = reader.outputs["LAI"] >> viewer.inputs["Grid"]

    composition.run(datetime(2000, 7, 1))
