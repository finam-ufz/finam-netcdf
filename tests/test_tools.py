import unittest

import xarray as xr
from finam import Composition, Info, Location, UniformGrid
from numpy.testing import assert_allclose

from finam_netcdf.tools import Layer, create_point_axis, extract_grid


class TestTools(unittest.TestCase):
    def test_read_grid(self):
        path = "tests/data/lai.nc"
        dataset = xr.open_dataset(path)
        layer = Layer(var="lai", xyz=("lon", "lat"), fixed={"time": 0})

        info, data = extract_grid(dataset, layer)

        self.assertTrue(isinstance(info.grid, UniformGrid))
        self.assertEqual(info.grid.data_location, Location.CELLS)

        self.assertEqual(
            info.grid.axes[0].shape[0], info.grid.data_axes[0].shape[0] + 1
        )
        self.assertEqual(
            info.grid.axes[1].shape[0], info.grid.data_axes[1].shape[0] + 1
        )

    def test_point_axis(self):
        cell_ax = [1, 2, 3, 4]
        point_ax = create_point_axis(cell_ax)
        self.assertEqual(len(point_ax), len(cell_ax) + 1)
        assert_allclose(point_ax, [0.5, 1.5, 2.5, 3.5, 4.5])

        cell_ax = [1, 3, 4]
        point_ax = create_point_axis(cell_ax)
        assert_allclose(point_ax, [0.0, 2.0, 3.5, 4.5])
