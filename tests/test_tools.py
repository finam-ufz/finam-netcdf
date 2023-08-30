import unittest

from finam import Location, RectilinearGrid
from netCDF4 import Dataset
from numpy.testing import assert_allclose

from finam_netcdf.tools import Layer, create_point_axis, extract_grid, extract_layers


class TestTools(unittest.TestCase):
    def test_read_grid(self):
        path = "tests/data/lai.nc"
        dataset = Dataset(path)
        time_var = "time"
        layer = Layer(var="lai", xyz=("lon", "lat"), fixed={"time": 0})

        info, data = extract_grid(dataset, layer, time_var)

        self.assertTrue(isinstance(info.grid, RectilinearGrid))
        self.assertEqual(info.grid.data_location, Location.CELLS)
        self.assertEqual(
            info.grid.axes[0].shape[0], info.grid.data_axes[1].shape[0] + 1
        )
        self.assertEqual(
            info.grid.axes[1].shape[0], info.grid.data_axes[0].shape[0] + 1
        )

    def test_point_axis(self):
        cell_ax = [1, 2, 3, 4]
        point_ax = create_point_axis(cell_ax)
        self.assertEqual(len(point_ax), len(cell_ax) + 1)
        assert_allclose(point_ax, [0.5, 1.5, 2.5, 3.5, 4.5])

        cell_ax = [1, 3, 4]
        point_ax = create_point_axis(cell_ax)
        assert_allclose(point_ax, [0.0, 2.0, 3.5, 4.5])

    def test_extract_layers(self):
        path = "tests/data/temp.nc"
        dataset = Dataset(path)
        time_var, layers = extract_layers(dataset)
        var_list = ["lon", "lat", "tmax", "tmin"]

        self.assertEqual(time_var, "time")

        is_present = layers[0].var in var_list
        assert is_present, f"{layers[0].var} is not present in the list"
        self.assertEqual(layers[0].xyz, ("xc", "yc"))

        is_present = layers[1].var in var_list
        assert is_present, f"{layers[1].var} is not present in the list"
        self.assertEqual(layers[1].xyz, ("xc", "yc"))

        is_present = layers[2].var in var_list
        assert is_present, f"{layers[2].var} is not present in the list"
        self.assertEqual(layers[2].xyz, ("xc", "yc"))

        is_present = layers[3].var in var_list
        assert is_present, f"{layers[3].var} is not present in the list"
        self.assertEqual(layers[3].xyz, ("xc", "yc"))
