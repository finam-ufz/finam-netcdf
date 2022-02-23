import unittest
import numpy as np
import xarray as xr

from datetime import datetime, timedelta

from finam.core.schedule import Composition
from finam.data.grid import Grid, GridSpec
from finam.modules.generators import CallbackGenerator

from finam_netcdf import Layer
from finam_netcdf.writer import NetCdfTimedWriter


def generate_grid():
    return Grid(GridSpec(10, 5), np.random.random(25))


class TestWriter(unittest.TestCase):
    def test_time_writer(self):
        path = "tests/out/test.nc"

        source1 = CallbackGenerator(
            callbacks={"Grid": lambda t: generate_grid()},
            start=datetime(2000, 1, 1),
            step=timedelta(days=1),
        )
        source2 = CallbackGenerator(
            callbacks={"Grid": lambda t: generate_grid()},
            start=datetime(2000, 1, 1),
            step=timedelta(days=1),
        )
        reader = NetCdfTimedWriter(
            path,
            {
                "LAI": Layer(var="lai", x="lon", y="lat"),
                "LAI2": Layer(var="lai2", x="lon", y="lat"),
            },
            time_var="time",
            start=datetime(2000, 1, 1),
            step=timedelta(days=1),
        )

        composition = Composition([source1, source2, reader])
        composition.initialize()

        _ = source1.outputs["Grid"] >> reader.inputs["LAI"]
        _ = source2.outputs["Grid"] >> reader.inputs["LAI2"]

        composition.run(datetime(2000, 1, 31))
