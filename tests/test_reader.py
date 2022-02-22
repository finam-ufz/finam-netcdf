import unittest
import xarray as xr

from finam_netcdf.reader import extract_grid, Layer


class TestReader(unittest.TestCase):
    def test_read_grid(self):
        path = "tests/data/lai.nc"
        dataset = xr.open_dataset(path)
        layer = Layer(var="lai", x="lon", y="lat", fixed={"time": 0})

        grid = extract_grid(dataset, layer)
