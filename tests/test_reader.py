import unittest
from datetime import datetime, timedelta

import finam as fm

from finam_netcdf import NetCdfReader, NetCdfStaticReader
from finam_netcdf.tools import Layer


class TestReader(unittest.TestCase):
    def test_init_reader(self):
        path = "tests/data/lai.nc"
        reader = NetCdfStaticReader(
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

        (reader.outputs["LAI"] >> consumer.inputs["Input"])

        comp.run(end_time=datetime(1901, 1, 2))

    def test_init_reader_no_time(self):
        path = "tests/data/temp.nc"
        reader = NetCdfStaticReader(
            path,
            {"Lat": Layer(var="lat", xyz=("xc", "yc"))},
        )
        consumer = fm.modules.DebugConsumer(
            {"Input": fm.Info(time=None, grid=None, units=None)},
            start=datetime(1901, 1, 1, 1, 0, 0),
            step=timedelta(days=1),
        )

        comp = fm.Composition([reader, consumer], log_level="DEBUG")
        comp.initialize()

        (reader.outputs["Lat"] >> consumer.inputs["Input"])

        comp.run(end_time=datetime(1901, 1, 2))

    def test_time_reader(self):
        path = "tests/data/lai.nc"
        reader = NetCdfReader(
            path,
            {
                "LAI": Layer(var="lai", xyz=("lon", "lat")),
                "LAI-stat": Layer(
                    var="lai", xyz=("lon", "lat"), fixed={"time": 0}, static=True
                ),
            },
            time_var="time",
        )

        consumer = fm.modules.DebugConsumer(
            {
                "Input": fm.Info(time=None, grid=None, units=None),
                "Input-stat": fm.Info(time=None, grid=None, units=None),
            },
            start=datetime(1901, 1, 1, 0, 1, 0),
            step=timedelta(minutes=1),
        )

        comp = fm.Composition([reader, consumer])
        comp.initialize()

        reader.outputs["LAI"] >> consumer.inputs["Input"]
        reader.outputs["LAI-stat"] >> consumer.inputs["Input-stat"]

        comp.connect()

        self.assertEqual(
            fm.data.get_magnitude(consumer.data["Input"][0, 0, 0]),
            fm.data.get_magnitude(consumer.data["Input-stat"][0, 0, 0]),
        )

        comp.run(end_time=datetime(1901, 1, 1, 0, 12))

        self.assertNotEqual(
            fm.data.get_magnitude(consumer.data["Input"][0, 0, 0]),
            fm.data.get_magnitude(consumer.data["Input-stat"][0, 0, 0]),
        )

    def test_time_reader_auto(self):
        path = "tests/data/lai.nc"
        reader = NetCdfReader(path)

        consumer = fm.modules.DebugConsumer(
            {
                "Input": fm.Info(time=None, grid=None, units=None),
            },
            start=datetime(1901, 1, 1, 0, 1, 0),
            step=timedelta(minutes=1),
        )

        comp = fm.Composition([reader, consumer])
        comp.initialize()

        reader.outputs["lai"] >> consumer.inputs["Input"]

        comp.run(end_time=datetime(1901, 1, 1, 0, 12))

    def test_time_reader_no_time(self):
        path = "tests/data/temp.nc"
        reader = NetCdfReader(
            path,
            {
                "Tmin": Layer(var="tmin", xyz=("xc", "yc")),
                "Lat": Layer(var="lat", xyz=("xc", "yc"), static=True),
            },
            time_var="time",
        )

        consumer = fm.modules.DebugConsumer(
            {
                "Input": fm.Info(time=None, grid=None, units=None),
            },
            start=datetime(1901, 1, 1, 0, 2, 0),
            step=timedelta(minutes=1),
        )

        comp = fm.Composition([reader, consumer])
        comp.initialize()

        reader.outputs["Lat"] >> consumer.inputs["Input"]

        comp.run(end_time=datetime(1901, 1, 1, 0, 12))

    def test_time_reader_limits(self):
        path = "tests/data/lai.nc"
        reader = NetCdfReader(
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

        comp.run(end_time=datetime(1901, 1, 1, 0, 12))

    def test_time_reader_callback(self):
        start = datetime(2000, 1, 1)
        step = timedelta(days=1)

        path = "tests/data/lai.nc"
        reader = NetCdfReader(
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

        comp.run(end_time=datetime(2000, 12, 31))
