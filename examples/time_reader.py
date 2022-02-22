from datetime import timedelta, datetime
import time

from finam.core.schedule import Composition
from finam.modules.visual import grid
from finam.modules import generators
from finam_netcdf.reader import Layer, NetCdfTimeReader

if __name__ == "__main__":
    path = "tests/data/lai.nc"

    reader = NetCdfTimeReader(
        path, {"LAI": Layer(var="lai", x="lon", y="lat")},
        time_var="time"
    )
    viewer = grid.GridView()

    sleep_mod = generators.CallbackGenerator(
        {"time": lambda t: time.sleep(0.5)},
        start=datetime(1901, 1, 1, 0, 1),
        step=timedelta(minutes=1),
    )

    composition = Composition([reader, viewer, sleep_mod])
    composition.initialize()

    _ = (
            reader.outputs()["LAI"]
            >> viewer.inputs()["Grid"]
    )

    composition.run(datetime(1901, 1, 1, 0, 12))
