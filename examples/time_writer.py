import os
from datetime import datetime, timedelta

import numpy as np
from finam import Composition, Info, UniformGrid
from finam.modules.generators import CallbackGenerator

from finam_netcdf import Layer
from finam_netcdf.writer import NetCdfTimedWriter


def random_grid(grid):
    return np.reshape(
        np.random.random(grid.data_size), newshape=grid.data_shape, order=grid.order
    )


if __name__ == "__main__":
    grid = UniformGrid((10, 5), data_location="POINTS")
    directory = "examples/output"
    if not os.path.exists(directory):
        os.mkdir(directory)

    file = os.path.join(directory, "test.nc")

    lai_gen = CallbackGenerator(
        callbacks={"LAI": (lambda t: random_grid(grid), Info(grid))},
        start=datetime(2000, 1, 1),
        step=timedelta(days=1),
    )
    sm_gen = CallbackGenerator(
        callbacks={"SM": (lambda t: random_grid(grid), Info(grid))},
        start=datetime(2000, 1, 1),
        step=timedelta(days=1),
    )
    writer = NetCdfTimedWriter(
        path=file,
        inputs={
            "LAI": Layer(var="lai", xyz=("x", "y")),
            "SM": Layer(var="soil_moisture", xyz=("x", "y")),
        },
        time_var="time",
        start=datetime(2000, 1, 1),
        step=timedelta(days=1),
    )

    composition = Composition([lai_gen, sm_gen, writer])
    composition.initialize()

    _ = lai_gen.outputs["LAI"] >> writer.inputs["LAI"]
    _ = sm_gen.outputs["SM"] >> writer.inputs["SM"]

    composition.run(datetime(2000, 1, 31))

    print("Wrote NetCDF file to %s" % (file,))
