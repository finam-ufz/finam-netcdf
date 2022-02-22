import unittest
import numpy as np
import xarray as xr

from finam_netcdf.reader import extract_grid, Layer


class TestReader(unittest.TestCase):
    def test_read_grid(self):
        path = "tests/data/lai.nc"
        dataset = xr.open_dataset(path)
        layer = Layer(var="lai", x="lon", y="lat", fixed={"time": 0})

        grid = extract_grid(dataset, layer)

        data = dataset["lai"]
        x = dataset.coords["lon"].data
        y = dataset.coords["lat"].data

        for _n in range(5):
            i, j = (
                np.random.randint(0, x.shape[0], 1)[0],
                np.random.randint(0, y.shape[0], 1)[0],
            )
            xx, yy = x[i], y[j]
            ci, cj = grid.to_cell(xx, yy)

            self.assertEqual(ci, i)
            self.assertEqual(cj, j)

            orig = data.isel({"time": 0, "lon": i, "lat": j}).data
            res = grid.get(i, j)
            self.assertEqual(orig, res)

            orig = data.sel(
                {"time": "1901-01-01T00:01:00.000000000", "lon": xx, "lat": yy}
            ).data
            res = grid.get(ci, cj)
            self.assertEqual(orig, res)
