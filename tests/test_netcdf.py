import unittest
import xarray as xr

from finam.data.grid import Grid, GridSpec


class TestNetCDF(unittest.TestCase):
    def test_read(self):
        path = "tests/data/lai.nc"
        dataset = xr.open_dataset(path)
        data = dataset.data_vars["lai"]

        # print(dataset.__class__)
        # print("============================")
        # print(data.__class__)
        # print(data[{"time": 0}])
        # print(data.coords["lon"].shape)
