import unittest
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from finam import FinamStatusError, UniformGrid, Composition, Info, UNITS
from finam.modules.debug import DebugConsumer

from finam_netcdf import Layer
from finam_netcdf.reader import NetCdfInitReader, NetCdfTimeReader, extract_grid


class TestReader(unittest.TestCase):
    def test_read_grid(self):
        path = "tests/data/lai.nc"
        dataset = xr.open_dataset(path)
        layer = Layer(var="lai", xyz=("lon", "lat"), fixed={"time": 0})

        info, data = extract_grid(dataset, layer)

        self.assertTrue(isinstance(info.grid, UniformGrid))

    def test_init_reader(self):
        path = "tests/data/lai.nc"
        reader = NetCdfInitReader(
            path, {"LAI": Layer(var="lai", xyz=("lon", "lat"), fixed={"time": 0})},
            time=datetime(1901, 1, 1, 0, 1, 0)
        )
        consumer = DebugConsumer(
            {"Input": Info(grid=None, units=None)},
            start=datetime(1900, 1, 1),
            step=timedelta(days=1),
        )

        comp = Composition([reader, consumer])
        comp.initialize()

        reader.outputs["LAI"] >> consumer.inputs["Input"]

        comp.run(datetime(1900, 1, 2))



    def test_time_reader(self):
        path = "tests/data/lai.nc"
        reader = NetCdfTimeReader(
            path, {"LAI": Layer(var="lai", x="lon", y="lat")}, time_var="time"
        )

        reader.initialize()
        reader.connect()

        _res = reader.outputs["LAI"].get_data(datetime(1901, 1, 1))

        self.assertEqual(reader.time, datetime(1901, 1, 1, 0, 1))
        # self.assertTrue(isinstance(res, Grid))

        reader.validate()

        reader.update()
        self.assertEqual(reader.time, datetime(1901, 1, 1, 0, 2))
        reader.update()
        self.assertEqual(reader.time, datetime(1901, 1, 1, 0, 3))
        reader.update()
        self.assertEqual(reader.time, datetime(1901, 1, 1, 0, 4))
        reader.update()
        self.assertEqual(reader.time, datetime(1901, 1, 1, 0, 5))

        reader.finalize()

    def test_time_reader_limits(self):
        path = "tests/data/lai.nc"
        reader = NetCdfTimeReader(
            path,
            {"LAI": Layer(var="lai", x="lon", y="lat")},
            time_var="time",
            time_limits=(datetime(1901, 1, 1, 0, 8), None),
        )

        reader.initialize()
        reader.connect()

        _res = reader.outputs["LAI"].get_data(datetime(1901, 1, 1))

        self.assertEqual(reader.time, datetime(1901, 1, 1, 0, 8))
        # self.assertTrue(isinstance(res, Grid))

        reader.validate()

        reader.update()
        self.assertEqual(reader.time, datetime(1901, 1, 1, 0, 9))
        reader.update()
        self.assertEqual(reader.time, datetime(1901, 1, 1, 0, 10))
        reader.update()
        self.assertEqual(reader.time, datetime(1901, 1, 1, 0, 11))
        reader.update()
        self.assertEqual(reader.time, datetime(1901, 1, 1, 0, 12))
        reader.update()

        with self.assertRaises(FinamStatusError):
            reader.update()

        reader.finalize()

    def test_time_reader_callback(self):
        start = datetime(2000, 1, 1)
        step = timedelta(days=1)

        path = "tests/data/lai.nc"
        reader = NetCdfTimeReader(
            path,
            {"LAI": Layer(var="lai", x="lon", y="lat")},
            time_var="time",
            time_callback=lambda s, _t, _i: (start + s * step, s % 12),
        )

        reader.initialize()
        reader.connect()

        _res = reader.outputs["LAI"].get_data(datetime(2000, 1, 1))

        self.assertEqual(reader.time, datetime(2000, 1, 1))
        # self.assertTrue(isinstance(res, Grid))

        reader.validate()

        for i in range(15):
            reader.update()
            self.assertEqual(reader.time, datetime(2000, 1, i + 2))
            _res = reader.outputs["LAI"].get_data(datetime(2000, 1, i + 2))
            # self.assertTrue(isinstance(res, Grid))

        reader.finalize()
