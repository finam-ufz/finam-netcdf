import os.path
import unittest
from datetime import datetime, timedelta
from os import path
from tempfile import TemporaryDirectory

import finam as fm
import numpy as np
from finam import Composition, Info, UniformGrid
from finam.components.generators import CallbackGenerator
from netCDF4 import Dataset

from finam_netcdf import NetCdfPushWriter, NetCdfStaticWriter, NetCdfTimedWriter


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

            # creating global attrs to the NetCDF output file - optional
            global_attrs = {
                "project_name": "test_time_writer",
                "original_source": "FINAM – Python model coupling framework",
                "creator_url": "https://finam.pages.ufz.de",
                "institution": "Helmholtz Centre for Environmental Research - UFZ (Helmholtz-Zentrum für Umweltforschung GmbH UFZ)",
                "description": "FINAM test: test_time_writer",
                "created_date": datetime.now().strftime("%d-%m-%Y"),
            }

            writer = NetCdfTimedWriter(
                path=file,
                inputs=["lai", "lai2"],
                step=timedelta(days=1),
                global_attrs=global_attrs,
            )

            composition = Composition([source1, source2, writer])

            source1.outputs["Grid"] >> writer.inputs["lai"]
            source2.outputs["Grid"] >> writer.inputs["lai2"]

            composition.run(end_time=datetime(2000, 1, 31))

            self.assertTrue(os.path.isfile(file))
            dataset = Dataset(file)

            lai = dataset["lai"]
            dims = list(lai.dimensions)

            self.assertEqual(dims, ["time", "x", "y"])
            self.assertEqual(lai.shape, (31, 10, 5))

            times = dataset["time"][:]
            self.assertEqual(times[0], 0.0)
            self.assertEqual(times[-1], 30.0)

            dataset.close()

    def test_static_writer(self):
        grid = UniformGrid((10, 5), data_location="POINTS")

        with TemporaryDirectory() as tmp:
            file = path.join(tmp, "test.nc")

            source = fm.components.StaticSimplexNoise(
                info=fm.Info(None, grid=grid, units="m"),
                frequency=0.05,
                octaves=3,
                persistence=0.5,
            )

            # creating global attrs to the NetCDF output file - optional
            global_attrs = {
                "project_name": "test_static_writer",
                "original_source": "FINAM – Python model coupling framework",
                "creator_url": "https://finam.pages.ufz.de",
                "institution": "Helmholtz Centre for Environmental Research - UFZ (Helmholtz-Zentrum für Umweltforschung GmbH UFZ)",
                "description": "FINAM test: test_static_writer",
                "created_date": datetime.now().strftime("%d-%m-%Y"),
            }

            writer = NetCdfStaticWriter(
                path=file,
                inputs=["height"],
                global_attrs=global_attrs,
            )

            composition = Composition([source, writer])

            source["Noise"] >> writer["height"]

            composition.run()

            self.assertTrue(os.path.isfile(file))
            dataset = Dataset(file)

            height = dataset["height"]
            dims = list(height.dimensions)

            self.assertEqual(dims, ["x", "y"])
            self.assertEqual(height.shape, (10, 5))

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
            writer = NetCdfPushWriter(path=file, inputs=["lai", "lai2"])

            composition = Composition([source1, source2, writer])

            source1.outputs["Grid"] >> writer.inputs["lai"]
            source2.outputs["Grid"] >> writer.inputs["lai2"]

            composition.run(end_time=datetime(2000, 1, 31))

            self.assertTrue(os.path.isfile(file))

            dataset = Dataset(file)
            lai = dataset["lai"]

            self.assertEqual(lai.dimensions, ("time", "x", "y"))

            times = dataset["time"][:]
            self.assertEqual(times[0], 0.0)
            self.assertEqual(times[-1], 30.0 * 86400)

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
            writer = NetCdfPushWriter(path=file, inputs=["lai", "lai2"])

            composition = Composition([source1, source2, writer])

            source1.outputs["Grid"] >> writer.inputs["lai"]
            source2.outputs["Grid"] >> writer.inputs["lai2"]

            with self.assertRaises(ValueError):
                composition.run(end_time=datetime(2000, 1, 31))


if __name__ == "__main__":
    unittest.main()
