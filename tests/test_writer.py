import unittest
from datetime import datetime, timedelta
from os import path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr
from finam.core.schedule import Composition
from finam.data.grid import Grid, GridSpec
from finam.modules.generators import CallbackGenerator

from finam_netcdf import Layer
from finam_netcdf.writer import NetCdfPushWriter, NetCdfTimedWriter


def generate_grid():
    return Grid(GridSpec(10, 5), data=np.random.random(50))


class TestWriter(unittest.TestCase):
    def test_time_writer(self):

        with TemporaryDirectory() as tmp:
            file = path.join(tmp, "test.nc")

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
            writer = NetCdfTimedWriter(
                path=file,
                inputs={
                    "LAI": Layer(var="lai", x="lon", y="lat"),
                    "LAI2": Layer(var="lai2", x="lon", y="lat"),
                },
                time_var="time",
                start=datetime(2000, 1, 1),
                step=timedelta(days=1),
            )

            composition = Composition([source1, source2, writer])
            composition.initialize()

            _ = source1.outputs["Grid"] >> writer.inputs["LAI"]
            _ = source2.outputs["Grid"] >> writer.inputs["LAI2"]

            composition.run(datetime(2000, 1, 31))

            dataset = xr.open_dataset(file)
            lai = dataset["lai"]

            self.assertEqual(lai.dims, ("time", "lat", "lon"))
            self.assertEqual(lai.coords["time"].shape, (30,))
            self.assertEqual(lai.coords["lat"].shape, (5,))
            self.assertEqual(lai.coords["lon"].shape, (10,))

            times = lai.coords["time"].data
            self.assertEqual(times[0], np.datetime64(datetime(2000, 1, 1)))
            self.assertEqual(times[-1], np.datetime64(datetime(2000, 1, 30)))

            dataset.close()

    def test_push_writer(self):

        with TemporaryDirectory() as tmp:
            file = path.join(tmp, "test.nc")

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
            writer = NetCdfPushWriter(
                path=file,
                inputs={
                    "LAI": Layer(var="lai", x="lon", y="lat"),
                    "LAI2": Layer(var="lai2", x="lon", y="lat"),
                },
                time_var="time",
            )

            composition = Composition([source1, source2, writer])
            composition.initialize()

            _ = source1.outputs["Grid"] >> writer.inputs["LAI"]
            _ = source2.outputs["Grid"] >> writer.inputs["LAI2"]

            composition.run(datetime(2000, 1, 31))

            dataset = xr.open_dataset(file)
            lai = dataset["lai"]

            self.assertEqual(lai.dims, ("time", "lat", "lon"))
            # TODO: push-based components receive one more time slice; document/fix?
            self.assertEqual(lai.coords["time"].shape, (31,))
            self.assertEqual(lai.coords["lat"].shape, (5,))
            self.assertEqual(lai.coords["lon"].shape, (10,))

            times = lai.coords["time"].data
            self.assertEqual(times[0], np.datetime64(datetime(2000, 1, 1)))
            self.assertEqual(times[-1], np.datetime64(datetime(2000, 1, 31)))

            dataset.close()

    def test_push_writer_fail(self):
        """
        Writer should fail if inputs have unequal time steps
        """

        with TemporaryDirectory() as tmp:
            file = path.join(tmp, "test.nc")

            source1 = CallbackGenerator(
                callbacks={"Grid": lambda t: generate_grid()},
                start=datetime(2000, 1, 1),
                step=timedelta(days=1),
            )
            source2 = CallbackGenerator(
                callbacks={"Grid": lambda t: generate_grid()},
                start=datetime(2000, 1, 1),
                step=timedelta(days=2),
            )
            writer = NetCdfPushWriter(
                path=file,
                inputs={
                    "LAI": Layer(var="lai", x="lon", y="lat"),
                    "LAI2": Layer(var="lai2", x="lon", y="lat"),
                },
                time_var="time",
            )

            composition = Composition([source1, source2, writer])
            composition.initialize()

            _ = source1.outputs["Grid"] >> writer.inputs["LAI"]
            _ = source2.outputs["Grid"] >> writer.inputs["LAI2"]

            with self.assertRaises(ValueError):
                composition.run(datetime(2000, 1, 31))
