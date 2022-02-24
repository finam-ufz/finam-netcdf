import os
import numpy as np
from datetime import timedelta, datetime

from finam.core.schedule import Composition
from finam.data.grid import Grid, GridSpec
from finam.modules.generators import CallbackGenerator
from finam_netcdf import Layer
from finam_netcdf.writer import NetCdfPushWriter


def random_grid():
    return Grid(GridSpec(20, 10), data=np.random.random(20 * 10))


if __name__ == "__main__":
    directory = "examples/output"
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    file = os.path.join(directory, "test.nc")

    lai_gen = CallbackGenerator(
        callbacks={"LAI": lambda t: random_grid()},
        start=datetime(2000, 1, 1),
        step=timedelta(days=1),
    )
    sm_gen = CallbackGenerator(
        callbacks={"SM": lambda t: random_grid()},
        start=datetime(2000, 1, 1),
        step=timedelta(days=1),
    )
    writer = NetCdfPushWriter(
        path=file,
        inputs={
            "LAI": Layer(var="lai", x="lon", y="lat"),
            "SM": Layer(var="soil_moisture", x="lon", y="lat"),
        },
        time_var="time"
    )

    composition = Composition([lai_gen, sm_gen, writer])
    composition.initialize()

    _ = lai_gen.outputs["LAI"] >> writer.inputs["LAI"]
    _ = sm_gen.outputs["SM"] >> writer.inputs["SM"]

    composition.run(datetime(2000, 1, 31))

    print("Wrote NetCDF file to %s" % (file,))
