from datetime import datetime

import finam as fm
from finam_plot import ContourPlot

from finam_netcdf import Layer
from finam_netcdf.reader import NetCdfReader

if __name__ == "__main__":
    start = datetime(2000, 1, 1)

    def to_time_step(tick, _last_time, _last_index):
        year = start.year + tick // 12
        month = 1 + tick % 12
        return datetime(year, month, 1), tick % 12

    path = "tests/data/lai.nc"

    reader = NetCdfReader(
        path,
        {"LAI": Layer(var="lai", xyz=("lon", "lat"))},
        time_var="time",
        time_callback=to_time_step,
    )

    viewer = ContourPlot()

    composition = fm.Composition([reader, viewer])
    composition.initialize()

    _ = reader.outputs["LAI"] >> fm.adapters.LinearTime() >> viewer.inputs["Grid"]

    composition.run(end_time=datetime(2005, 1, 1))
