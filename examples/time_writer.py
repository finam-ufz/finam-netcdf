import os
from datetime import datetime, timedelta

import finam as fm
import numpy as np

from finam_netcdf import NetCdfTimedWriter


def random_grid(grid):
    return np.reshape(
        np.random.random(grid.data_size), newshape=grid.data_shape, order=grid.order
    )


grid = fm.UniformGrid((10, 5), axes_increase=(True, False), data_location="POINTS")
directory = "examples/output"
if not os.path.exists(directory):
    os.mkdir(directory)

file = os.path.join(directory, "test.nc")

lai_gen = fm.components.SimplexNoise(
    info=fm.Info(datetime(2000, 1, 1), grid=grid, units="m"),
    frequency=0.05,
    time_frequency=1.0 / (30 * 24 * 3600),
    octaves=3,
    persistence=0.5,
)
sm_gen = fm.components.CallbackGenerator(
    callbacks={"SM": (lambda t: random_grid(grid), fm.Info(None, grid))},
    start=datetime(2000, 1, 1),
    step=timedelta(days=1),
)
writer = NetCdfTimedWriter(file, inputs=["lai", "SM"], step=timedelta(days=1))

composition = fm.Composition([lai_gen, sm_gen, writer])

lai_gen.outputs["Noise"] >> writer.inputs["lai"]
sm_gen.outputs["SM"] >> writer.inputs["SM"]

composition.run(end_time=datetime(2000, 1, 31))

print("Wrote NetCDF file to %s" % (file,))
