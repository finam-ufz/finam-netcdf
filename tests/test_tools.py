import unittest
from os import path
from tempfile import TemporaryDirectory

import finam as fm
import numpy as np
from finam import Location, RectilinearGrid
from netCDF4 import Dataset
from numpy.testing import assert_allclose

from finam_netcdf.tools import (
    Variable,
    _create_point_axis,
    extract_data,
    extract_info,
    extract_time,
    extract_variables,
    set_mask,
)


class TestTools(unittest.TestCase):
    def _create_masked_dataset(self, file_path):
        dataset = Dataset(file_path, "w")
        dataset.createDimension("x", 3)
        var = dataset.createVariable("v", "f4", ("x",), fill_value=-999.0)
        var[:] = np.array([1.0, -999.0, 3.0], dtype="f4")
        dataset.close()

    def test_read_grid(self):
        path = "tests/data/lai.nc"
        dataset = Dataset(path)
        time_var = "time"
        variable = Variable("lai", slices={"time": 0}, mask=fm.Mask.FLEX)

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

    def test_set_mask_flex(self):
        with TemporaryDirectory() as tmp:
            file_path = path.join(tmp, "mask.nc")
            self._create_masked_dataset(file_path)
            dataset = Dataset(file_path)
            variable = Variable("v", static=True, mask=fm.Mask.FLEX)
            info = fm.Info(time=None, grid=None)
            data = extract_data(dataset, variable)
            self.assertTrue(np.ma.isMaskedArray(data))

            result = set_mask(info, data, dataset, variable)
            self.assertTrue(np.ma.isMaskedArray(result))
            self.assertTrue(result.mask[1])
            self.assertEqual(info.mask, fm.Mask.FLEX)
            dataset.close()

    def test_set_mask_none(self):
        with TemporaryDirectory() as tmp:
            file_path = path.join(tmp, "mask.nc")
            self._create_masked_dataset(file_path)
            dataset = Dataset(file_path)
            variable = Variable("v", static=True, mask=None)
            info = fm.Info(time=None, grid=None)
            data = extract_data(dataset, variable)
            result = set_mask(info, data, dataset, variable)

            self.assertTrue(np.ma.isMaskedArray(result))
            self.assertTrue(np.array_equal(variable.mask, data.mask))
            self.assertTrue(np.array_equal(info.mask, data.mask))
            dataset.close()

    def test_set_mask_none_masked_output(self):
        with TemporaryDirectory() as tmp:
            file_path = path.join(tmp, "mask.nc")
            self._create_masked_dataset(file_path)
            dataset = Dataset(file_path)
            variable = Variable("v", static=True, mask=fm.Mask.NONE)
            info = fm.Info(time=None, grid=None)
            data = extract_data(dataset, variable)
            result = set_mask(info, data, dataset, variable)

            self.assertFalse(np.ma.isMaskedArray(result))
            self.assertEqual(result[1], -999.0)
            dataset.close()

    def test_set_mask_invalid(self):
        with TemporaryDirectory() as tmp:
            file_path = path.join(tmp, "mask.nc")
            self._create_masked_dataset(file_path)
            dataset = Dataset(file_path)
            variable = Variable("v", static=True, mask=np.array([True, False, False]))
            info = fm.Info(time=None, grid=None)
            data = extract_data(dataset, variable)
            with self.assertRaises(ValueError):
                set_mask(info, data, dataset, variable)
            dataset.close()


if __name__ == "__main__":
    unittest.main()
