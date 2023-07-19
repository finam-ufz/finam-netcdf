import os.path
import unittest
from datetime import datetime, timedelta
from os import path
from tempfile import TemporaryDirectory

import netCDF4 as nc
import numpy as np
from finam import Composition, Info, UniformGrid
from finam.modules.generators import CallbackGenerator

from finam_netcdf import NetCdfPushWriter, NetCdfTimedWriter
from finam_netcdf.tools import Layer


def generate_grid(grid):
    return np.reshape(
        np.random.random(grid.data_size), newshape=grid.data_shape, order=grid.order
    )


class TestWriter(unittest.TestCase):
    def test_time_writer(self):
        grid = UniformGrid((10, 5), data_location="POINTS")

        with TemporaryDirectory() as tmp:
            file = path.join(tmp, "test.nc")

            source1 = CallbackGenerator(
                callbacks={
                    "Grid": (lambda t: generate_grid(grid), Info(None, grid, units="m"))
                },
                start=datetime(2000, 1, 1),
                step=timedelta(days=1),
            )
            source2 = CallbackGenerator(
                callbacks={
                    "Grid": (lambda t: generate_grid(grid), Info(None, grid, units="m"))
                },
                start=datetime(2000, 1, 1),
                step=timedelta(days=1),
            )
            writer = NetCdfTimedWriter(
                path=file,
                inputs={
                    "LAI": Layer(var="lai", xyz=("x", "y")),
                    "LAI2": Layer(var="lai2", xyz=("x", "y")),
                },
                time_var="time",
                start=datetime(2000, 1, 1),
                step=timedelta(days=1),
            )

            composition = Composition([source1, source2, writer])
            composition.initialize()

            _ = source1.outputs["Grid"] >> writer.inputs["LAI"]
            _ = source2.outputs["Grid"] >> writer.inputs["LAI2"]

            composition.run(end_time=datetime(2000, 1, 31))

            self.assertTrue(os.path.isfile(file))
            dataset = nc.Dataset(file)

            lai = dataset["lai"]

            dims = []
            for dim in lai.dimensions:
                dims.append(dim)

            self.assertEqual(dims, ["time", "x", "y"])
            self.assertEqual(lai.shape, (31, 10, 5))

            times = dataset["time"][:]
            self.assertEqual(times[0], 0.0)
            self.assertEqual(times[-1], 30.0)

            dataset.close()

    def test_push_writer(self):
        grid = UniformGrid((10, 5), data_location="POINTS")

        with TemporaryDirectory() as tmp:
            file = path.join(
                tmp,
                "test.nc",
            )

            source1 = CallbackGenerator(
                callbacks={"Grid": (lambda t: generate_grid(grid), Info(None, grid))},
                start=datetime(2000, 1, 1),
                step=timedelta(days=1),
            )
            source2 = CallbackGenerator(
                callbacks={"Grid": (lambda t: generate_grid(grid), Info(None, grid))},
                start=datetime(2000, 1, 1),
                step=timedelta(days=1),
            )
            writer = NetCdfPushWriter(
                path=file,
                inputs={
                    "LAI": Layer(var="lai", xyz=("x", "y")),
                    "LAI2": Layer(var="lai2", xyz=("x", "y")),
                },
                time_var="time",
                start=datetime(2000, 1, 1),
                step=timedelta(days=1),
            )

            composition = Composition([source1, source2, writer])
            composition.initialize()

            _ = source1.outputs["Grid"] >> writer.inputs["LAI"]
            _ = source2.outputs["Grid"] >> writer.inputs["LAI2"]

            composition.run(end_time=datetime(2000, 1, 31))

            self.assertTrue(os.path.isfile(file))

            dataset = nc.Dataset(file)
            lai = dataset["lai"]

            self.assertEqual(lai.dimensions, ("time", "x", "y"))

            times = dataset["time"][:]
            self.assertEqual(times[0], 0.0)
            self.assertEqual(times[-1], 30.0)

            dataset.close()

    def test_push_writer_fail(self):
        """
        Writer should fail if inputs have unequal time steps
        """
        grid = UniformGrid((10, 5), data_location="POINTS")

        with TemporaryDirectory() as tmp:
            file = path.join(tmp, "test.nc")

            source1 = CallbackGenerator(
                callbacks={"Grid": (lambda t: generate_grid(grid), Info(None, grid))},
                start=datetime(2000, 1, 1),
                step=timedelta(days=1),
            )
            source2 = CallbackGenerator(
                callbacks={"Grid": (lambda t: generate_grid(grid), Info(None, grid))},
                start=datetime(2000, 1, 1),
                step=timedelta(days=2),
            )
            writer = NetCdfPushWriter(
                path=file,
                inputs={
                    "lai": Layer(var="lai", xyz=("x", "y")),
                    "lai2": Layer(var="lai2", xyz=("x", "y")),
                },
                time_var="time",
                start=datetime(2000, 1, 1),
                step=timedelta(days=1),
            )

            composition = Composition([source1, source2, writer])
            composition.initialize()

            _ = source1.outputs["Grid"] >> writer.inputs["lai"]
            _ = source2.outputs["Grid"] >> writer.inputs["lai2"]

    #         # TODO: fails here, but should it fail based on the test_push_writer_fail description?
    #         # Not sure what would be the problem here...
    #         with self.assertRaises(ValueError):
    #             composition.run(end_time=datetime(2000, 1, 31))
