import unittest
from datetime import datetime, timedelta

import xarray as xr
from finam import Composition, Info, UniformGrid
from finam.modules.debug import DebugConsumer

from finam_netcdf import NetCdfInitReader, NetCdfTimeReader
from finam_netcdf.tools import Layer, extract_grid


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
            path,
            {"LAI": Layer(var="lai", xyz=("lon", "lat"), fixed={"time": 0})},
        )
        consumer = DebugConsumer(
            {"Input": Info(grid=None, units=None)},
            start=datetime(1901, 1, 1),
            step=timedelta(days=1),
        )

        comp = Composition([reader, consumer], log_level="DEBUG")
        comp.initialize()

        reader.outputs["LAI"] >> consumer.inputs["Input"]

        comp.run(datetime(1901, 1, 2))

    def test_time_reader(self):
        path = "tests/data/lai.nc"
        reader = NetCdfTimeReader(
            path, {"LAI": Layer(var="lai", xyz=("lon", "lat"))}, time_var="time"
        )

        consumer = DebugConsumer(
            {"Input": Info(grid=None, units=None)},
            start=datetime(1901, 1, 1, 0, 1),
            step=timedelta(hours=1),
        )

        comp = Composition([reader, consumer], log_level="DEBUG")
        comp.initialize()

        reader.outputs["LAI"] >> consumer.inputs["Input"]

        comp.run(datetime(1901, 1, 1, 0, 12))

    def test_time_reader_limits(self):
        path = "tests/data/lai.nc"
        reader = NetCdfTimeReader(
            path,
            {"LAI": Layer(var="lai", xyz=("lon", "lat"))},
            time_var="time",
            time_limits=(datetime(1901, 1, 1, 0, 8), None),
        )

        consumer = DebugConsumer(
            {"Input": Info(grid=None, units=None)},
            start=datetime(1901, 1, 1, 0, 1),
            step=timedelta(hours=1),
        )

        comp = Composition([reader, consumer], log_level="DEBUG")
        comp.initialize()

        reader.outputs["LAI"] >> consumer.inputs["Input"]

        comp.run(datetime(1901, 1, 1, 0, 12))

    def test_time_reader_callback(self):
        start = datetime(2000, 1, 1)
        step = timedelta(days=1)

        path = "tests/data/lai.nc"
        reader = NetCdfTimeReader(
            path,
            {"LAI": Layer(var="lai", xyz=("lon", "lat"))},
            time_var="time",
            time_callback=lambda s, _t, _i: (start + s * step, s % 12),
        )

        consumer = DebugConsumer(
            {"Input": Info(grid=None, units=None)},
            start=datetime(2000, 1, 1),
            step=timedelta(days=1),
        )

        comp = Composition([reader, consumer], log_level="DEBUG")
        comp.initialize()

        reader.outputs["LAI"] >> consumer.inputs["Input"]

        comp.run(datetime(2000, 12, 31))
