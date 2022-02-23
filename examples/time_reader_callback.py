from datetime import timedelta, datetime
import time

from finam.adapters.time import LinearInterpolation
from finam.core.schedule import Composition
from finam.modules.visual import grid
from finam.modules import generators
from finam_netcdf.reader import Layer, NetCdfTimeReader

if __name__ == "__main__":
    start = datetime(2000, 1, 1)
    step = timedelta(days=30)

    path = "tests/data/lai.nc"

    reader = NetCdfTimeReader(
        path,
        {"LAI": Layer(var="lai", x="lon", y="lat")},
        time_var="time",
        time_callback=lambda s, _t, _i: (start + s * step, s % 12),
    )

    viewer = grid.TimedGridView(start=start, step=timedelta(days=6))

    composition = Composition([reader, viewer])
    composition.initialize()

    _ = reader.outputs()["LAI"] >> LinearInterpolation() >> viewer.inputs()["Grid"]

    composition.run(datetime(2005, 1, 1))
