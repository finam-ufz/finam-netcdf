import unittest

from finam import Location, RectilinearGrid
from netCDF4 import Dataset
from numpy.testing import assert_allclose

from finam_netcdf.tools import (
    Variable,
    _create_point_axis,
    extract_info,
    extract_time,
    extract_variables,
)


class TestTools(unittest.TestCase):
    def test_read_grid(self):
        path = "tests/data/lai.nc"
        dataset = Dataset(path)
        time_var = "time"
        variable = Variable("lai", slices={"time": 0})

        info = extract_info(dataset, variable)
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
        point_ax = _create_point_axis(cell_ax)
        self.assertEqual(len(point_ax), len(cell_ax) + 1)
        assert_allclose(point_ax, [0.5, 1.5, 2.5, 3.5, 4.5])

        cell_ax = [1, 3, 4]
        point_ax = _create_point_axis(cell_ax)
        assert_allclose(point_ax, [0.0, 2.0, 3.5, 4.5])

    def test_extract_variables(self):
        path = "tests/data/temp.nc"
        dataset = Dataset(path)
        variables = extract_variables(dataset)
        var_list = ["lon", "lat", "tmax", "tmin"]
        time_var = extract_time(dataset)
        self.assertEqual(time_var, "time")

        self.assertTrue(len(variables) == 4)
        self.assertTrue(variables[0].name in var_list)
        self.assertTrue(variables[1].name in var_list)
        self.assertTrue(variables[2].name in var_list)
        self.assertTrue(variables[3].name in var_list)
        for var in variables:
            if var.name in ["lon", "lat"]:
                self.assertTrue(var.static)
            else:
                self.assertFalse(var.static)


if __name__ == "__main__":
    unittest.main()
