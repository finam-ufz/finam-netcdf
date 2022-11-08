import unittest
from datetime import datetime, timedelta

import finam as fm

from finam_netcdf import NetCdfInitReader, NetCdfTimeReader
from finam_netcdf.tools import Layer


class TestReader(unittest.TestCase):
    def test_init_reader(self):
        path = "tests/data/lai.nc"
        reader = NetCdfInitReader(
            path,
            {"LAI": Layer(var="lai", xyz=("lon", "lat"), fixed={"time": 0})},
        )
        consumer = fm.modules.DebugConsumer(
            {"Input": fm.Info(time=None, grid=None, units=None)},
            start=datetime(1901, 1, 1, 1, 0, 0),
            step=timedelta(days=1),
        )

        comp = fm.Composition([reader, consumer], log_level="DEBUG")
        comp.initialize()

        (
            reader.outputs["LAI"]
            >> fm.adapters.ExtrapolateTime()
            >> consumer.inputs["Input"]
        )

        comp.run(datetime(1901, 1, 2))

    def test_time_reader(self):
        path = "tests/data/lai.nc"
        reader = NetCdfTimeReader(
            path, {"LAI": Layer(var="lai", xyz=("lon", "lat"))}, time_var="time"
        )

        consumer = fm.modules.DebugConsumer(
            {"Input": fm.Info(time=None, grid=None, units=None)},
            start=datetime(1901, 1, 1, 0, 1, 0),
            step=timedelta(minutes=1),
        )

        comp = fm.Composition([reader, consumer], log_level="DEBUG")
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

        consumer = fm.modules.DebugConsumer(
            {"Input": fm.Info(time=None, grid=None, units=None)},
            start=datetime(1901, 1, 1, 0, 8),
            step=timedelta(minutes=1),
        )

        comp = fm.Composition([reader, consumer], log_level="DEBUG")
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

        consumer = fm.modules.DebugConsumer(
            {"Input": fm.Info(time=None, grid=None, units=None)},
            start=datetime(2000, 1, 1),
            step=timedelta(days=1),
        )

        comp = fm.Composition([reader, consumer], log_level="DEBUG")
        comp.initialize()

        reader.outputs["LAI"] >> consumer.inputs["Input"]

        comp.run(datetime(2000, 12, 31))
