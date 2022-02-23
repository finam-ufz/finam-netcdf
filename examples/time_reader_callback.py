from datetime import timedelta, datetime

from finam.adapters.time import LinearInterpolation
from finam.core.schedule import Composition
from finam.modules.visual import grid
from finam_netcdf.reader import Layer, NetCdfTimeReader

if __name__ == "__main__":
    start = datetime(2000, 1, 1)

    def to_time_step(tick, _last_time, _last_index):
        year = start.year + tick // 12
        month = 1 + tick % 12
        return datetime(year, month, 1), tick % 12

    path = "tests/data/lai.nc"

    reader = NetCdfTimeReader(
        path,
        {"LAI": Layer(var="lai", x="lon", y="lat")},
        time_var="time",
        time_callback=to_time_step,
    )

    viewer = grid.TimedGridView(start=start, step=timedelta(days=6))

    composition = Composition([reader, viewer])
    composition.initialize()

    _ = reader.outputs()["LAI"] >> LinearInterpolation() >> viewer.inputs()["Grid"]

    composition.run(datetime(2005, 1, 1))
