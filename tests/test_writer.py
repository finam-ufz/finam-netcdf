import os.path
import unittest
from datetime import datetime, timedelta
from os import path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr
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

            dataset = xr.open_dataset(file)
            lai = dataset["lai"]

            self.assertEqual(lai.dims, ("time", "y", "x"))
            self.assertEqual(lai.coords["time"].shape, (31,))
            self.assertEqual(lai.coords["x"].shape, (10,))
            self.assertEqual(lai.coords["y"].shape, (5,))

            times = lai.coords["time"].data
            self.assertEqual(times[0], np.datetime64(datetime(2000, 1, 1)))
            self.assertEqual(times[-1], np.datetime64(datetime(2000, 1, 31)))

            dataset.close()

    def test_push_writer(self):
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
                step=timedelta(days=1),
            )
            writer = NetCdfPushWriter(
                path=file,
                inputs={
                    "LAI": Layer(var="lai", xyz=("x", "y")),
                    "LAI2": Layer(var="lai2", xyz=("x", "y")),
                },
                time_var="time",
            )

            composition = Composition([source1, source2, writer])
            composition.initialize()

            _ = source1.outputs["Grid"] >> writer.inputs["LAI"]
            _ = source2.outputs["Grid"] >> writer.inputs["LAI2"]

            composition.run(end_time=datetime(2000, 1, 31))

            self.assertTrue(os.path.isfile(file))

            dataset = xr.open_dataset(file)
            lai = dataset["lai"]

            self.assertEqual(lai.dims, ("time", "y", "x"))
            self.assertEqual(lai.coords["time"].shape, (31,))
            self.assertEqual(lai.coords["x"].shape, (10,))
            self.assertEqual(lai.coords["y"].shape, (5,))

            times = lai.coords["time"].data
            self.assertEqual(times[0], np.datetime64(datetime(2000, 1, 1)))
            self.assertEqual(times[-1], np.datetime64(datetime(2000, 1, 31)))

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
                    "LAI": Layer(var="lai", xyz=("x", "y")),
                    "LAI2": Layer(var="lai2", xyz=("x", "y")),
                },
                time_var="time",
            )

            composition = Composition([source1, source2, writer])
            composition.initialize()

            _ = source1.outputs["Grid"] >> writer.inputs["LAI"]
            _ = source2.outputs["Grid"] >> writer.inputs["LAI2"]

            with self.assertRaises(ValueError):
                composition.run(end_time=datetime(2000, 1, 31))
