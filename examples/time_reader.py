from datetime import timedelta, datetime

from finam.adapters.time import LinearInterpolation
from finam.core.schedule import Composition
from finam.modules.visual import grid
from finam_netcdf.reader import Layer, NetCdfTimeReader

if __name__ == "__main__":
    path = "tests/data/lai.nc"

    reader = NetCdfTimeReader(
        path, {"LAI": Layer(var="lai", x="lon", y="lat")}, time_var="time"
    )

    viewer = grid.TimedGridView(
        start=datetime(1901, 1, 1, 0, 1), step=timedelta(seconds=5)
    )

    composition = Composition([reader, viewer])
    composition.initialize()

    _ = reader.outputs()["LAI"] >> LinearInterpolation() >> viewer.inputs()["Grid"]

    composition.run(datetime(1901, 1, 1, 0, 12))
