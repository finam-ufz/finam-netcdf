import unittest
from datetime import datetime, timedelta

import finam as fm
import numpy as np
from netCDF4 import Dataset

from finam_netcdf import NetCdfReader, NetCdfStaticReader, Variable


class TestReader(unittest.TestCase):
    def test_init_reader(self):
        path = "tests/data/lai.nc"
        reader = NetCdfStaticReader(path, [Variable("lai", slices={"time": 0})])
        consumer = fm.components.DebugConsumer(
            {"Input": fm.Info(time=None, grid=None, units=None)},
            start=datetime(1901, 1, 1, 1, 0, 0),
            step=timedelta(days=1),
        )

        comp = fm.Composition([reader, consumer], log_level="DEBUG")

        (reader.outputs["lai"] >> consumer.inputs["Input"])

        comp.run(end_time=datetime(1901, 1, 2))

    def test_init_reader_no_time(self):
        path = "tests/data/temp.nc"
        reader = NetCdfStaticReader(path, ["lat"])
        consumer = fm.components.DebugConsumer(
            {"Input": fm.Info(time=None, grid=None, units=None)},
            start=datetime(1901, 1, 1, 1, 0, 0),
            step=timedelta(days=1),
        )

        comp = fm.Composition([reader, consumer], log_level="DEBUG")

        (reader.outputs["lat"] >> consumer.inputs["Input"])

        comp.run(end_time=datetime(1901, 1, 2))

    def test_time_reader(self):
        path = "tests/data/lai.nc"
        reader = NetCdfReader(
            path, ["lai", Variable("lai", io_name="LAI-stat", slices={"time": 0})]
        )

        consumer = fm.components.DebugConsumer(
            {
                "Input": fm.Info(time=None, grid=None, units=None),
                "Input-stat": fm.Info(time=None, grid=None, units=None),
            },
            start=datetime(1901, 1, 1, 0, 1, 0),
            step=timedelta(minutes=1),
        )

        comp = fm.Composition([reader, consumer])

        reader.outputs["lai"] >> consumer.inputs["Input"]
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

        consumer = fm.components.DebugConsumer(
            {
                "Input": fm.Info(time=None, grid=None, units=None),
            },
            start=datetime(1901, 1, 1, 0, 1, 0),
            step=timedelta(minutes=1),
        )

        comp = fm.Composition([reader, consumer])

        reader.outputs["lai"] >> consumer.inputs["Input"]

        comp.run(end_time=datetime(1901, 1, 1, 0, 12))

    def test_time_reader_no_time(self):
        path = "tests/data/temp.nc"
        reader = NetCdfReader(path, ["tmin", "lat"])

        consumer = fm.components.DebugConsumer(
            {
                "Input": fm.Info(time=None, grid=None, units=None),
            },
            start=datetime(1901, 1, 1, 0, 2, 0),
            step=timedelta(minutes=1),
        )

        comp = fm.Composition([reader, consumer])
        reader.outputs["lat"] >> consumer.inputs["Input"]

        comp.run(end_time=datetime(1901, 1, 1, 0, 12))

    def test_time_reader_limits(self):
        path = "tests/data/lai.nc"
        reader = NetCdfReader(
            path, ["lai"], time_limits=(datetime(1901, 1, 1, 0, 8), None)
        )

        consumer = fm.components.DebugConsumer(
            {"Input": fm.Info(time=None, grid=None, units=None)},
            start=datetime(1901, 1, 1, 0, 8),
            step=timedelta(minutes=1),
        )

        comp = fm.Composition([reader, consumer], log_level="DEBUG")

        reader.outputs["lai"] >> consumer.inputs["Input"]

        comp.run(end_time=datetime(1901, 1, 1, 0, 12))

    def test_time_reader_callback(self):
        start = datetime(2000, 1, 1)
        step = timedelta(days=1)

        path = "tests/data/lai.nc"
        reader = NetCdfReader(
            path, ["lai"], time_callback=lambda s, _t, _i: (start + s * step, s % 12)
        )

        consumer = fm.components.DebugConsumer(
            {"Input": fm.Info(time=None, grid=None, units=None)},
            start=datetime(2000, 1, 1),
            step=timedelta(days=1),
        )

        comp = fm.Composition([reader, consumer], log_level="DEBUG")

        reader.outputs["lai"] >> consumer.inputs["Input"]

        comp.run(end_time=datetime(2000, 12, 31))

    def test_time_reader_crs(self):
        path = "tests/data/with_crs.nc"
        reader = NetCdfReader(path)

        consumer = fm.components.DebugConsumer(
            {
                "lai1": fm.Info(time=None, grid=None, units=None),
                "lai2": fm.Info(time=None, grid=None, units=None),
                "lai3": fm.Info(time=None, grid=None, units=None),
            },
            start=datetime(2000, 1, 1, 0, 0, 0),
            step=timedelta(days=1),
        )

        comp = fm.Composition([reader, consumer], log_level="DEBUG")

        reader.outputs["lai"] >> consumer.inputs["lai1"]
        reader.outputs["lai2"] >> consumer.inputs["lai2"]
        reader.outputs["lai3"] >> consumer.inputs["lai3"]

        comp.connect()

        comp.run(end_time=datetime(2000, 1, 31, 0, 0))

        self.assertEqual(
            consumer.inputs["lai1"].info.grid.crs.name, "ETRS89-extended / LAEA Europe"
        )
        self.assertEqual(consumer.inputs["lai2"].info.grid.crs.name, "WGS 84")
        self.assertEqual(consumer.inputs["lai3"].info.grid.crs, None)


class TestMeshReader(unittest.TestCase):
    def _mesh_meta(self, path):
        with Dataset(path) as dataset:
            mesh_vars = [
                name
                for name in dataset.variables
                if getattr(dataset[name], "cf_role", None) == "mesh_topology"
            ]
            self.assertEqual(len(mesh_vars), 1)
            mesh_name = mesh_vars[0]
            mesh_var = dataset[mesh_name]
            node_coords = mesh_var.node_coordinates.split()
            n_nodes = dataset.variables[node_coords[0]].shape[0]
            n_cells = dataset.variables["cell_value_static"].shape[0]
            conn_name = None
            for attr in (
                "edge_node_connectivity",
                "face_node_connectivity",
                "volume_node_connectivity",
            ):
                if attr in mesh_var.ncattrs():
                    conn_name = getattr(mesh_var, attr)
                    break
            n_conn_cols = (
                dataset.variables[conn_name].shape[1] if conn_name is not None else None
            )
        return node_coords, n_nodes, n_cells, n_conn_cols

    def _assert_mesh_grid(self, path, node_grid, cell_grid):
        node_coords, n_nodes, n_cells, n_conn_cols = self._mesh_meta(path)

        self.assertIsInstance(node_grid, fm.UnstructuredPoints)
        self.assertIsInstance(cell_grid, fm.UnstructuredGrid)
        self.assertEqual(node_grid.data_location, fm.Location.POINTS)
        self.assertEqual(cell_grid.data_location, fm.Location.CELLS)

        self.assertEqual(node_grid.points.shape, (n_nodes, len(node_coords)))
        self.assertEqual(cell_grid.points.shape, (n_nodes, len(node_coords)))
        self.assertEqual(cell_grid.cells.shape[0], n_cells)
        if n_conn_cols is not None:
            self.assertEqual(cell_grid.cells.shape[1], n_conn_cols)

    def _assert_cell_types(self, path, cell_grid):
        cell_types = list(map(int, cell_grid.cell_types))
        expected = {
            "tests/data/mesh1d.nc": [int(fm.CellType.LINE)],
            "tests/data/mesh2d.nc": [int(fm.CellType.TRI), int(fm.CellType.QUAD)],
            "tests/data/mesh3d.nc": [int(fm.CellType.TETRA), int(fm.CellType.HEX)],
        }[path]
        if len(expected) == 1:
            self.assertTrue(all(ct == expected[0] for ct in cell_types))
        else:
            self.assertEqual(cell_types, expected)

    def test_static_mesh_reader(self):
        for path in (
            "tests/data/mesh1d.nc",
            "tests/data/mesh2d.nc",
            "tests/data/mesh3d.nc",
        ):
            with self.subTest(path=path):
                reader = NetCdfStaticReader(
                    path,
                    [Variable("node_value_static"), Variable("cell_value_static")],
                )
                consumer = fm.components.DebugConsumer(
                    {
                        "node": fm.Info(time=None, grid=None, units=None),
                        "cell": fm.Info(time=None, grid=None, units=None),
                    },
                    start=datetime(2025, 1, 1),
                    step=timedelta(hours=1),
                )
                comp = fm.Composition([reader, consumer])
                reader.outputs["node_value_static"] >> consumer.inputs["node"]
                reader.outputs["cell_value_static"] >> consumer.inputs["cell"]
                comp.run(end_time=datetime(2025, 1, 1, 1))

                node_grid = consumer.inputs["node"].info.grid
                cell_grid = consumer.inputs["cell"].info.grid
                self._assert_mesh_grid(path, node_grid, cell_grid)
                self._assert_cell_types(path, cell_grid)

                node_data = fm.data.get_magnitude(consumer.data["node"])
                cell_data = fm.data.get_magnitude(consumer.data["cell"])
                _, n_nodes, n_cells, _ = self._mesh_meta(path)
                self.assertEqual(node_data.shape[-1], n_nodes)
                self.assertEqual(cell_data.shape[-1], n_cells)

    def test_time_mesh_reader(self):
        for path in (
            "tests/data/mesh1d.nc",
            "tests/data/mesh2d.nc",
            "tests/data/mesh3d.nc",
        ):
            with self.subTest(path=path):
                reader = NetCdfReader(path, ["node_value", "cell_value"])
                consumer = fm.components.DebugConsumer(
                    {
                        "node": fm.Info(time=None, grid=None, units=None),
                        "cell": fm.Info(time=None, grid=None, units=None),
                    },
                    start=datetime(2025, 1, 1),
                    step=timedelta(hours=1),
                )
                comp = fm.Composition([reader, consumer])
                reader.outputs["node_value"] >> consumer.inputs["node"]
                reader.outputs["cell_value"] >> consumer.inputs["cell"]

                comp.connect()
                node0 = fm.data.get_magnitude(consumer.data["node"]).copy()
                cell0 = fm.data.get_magnitude(consumer.data["cell"]).copy()

                node_grid = consumer.inputs["node"].info.grid
                cell_grid = consumer.inputs["cell"].info.grid
                self._assert_mesh_grid(path, node_grid, cell_grid)
                self._assert_cell_types(path, cell_grid)

                comp.run(end_time=datetime(2025, 1, 1, 2))

                node1 = fm.data.get_magnitude(consumer.data["node"]).copy()
                cell1 = fm.data.get_magnitude(consumer.data["cell"]).copy()
                self.assertTrue(np.any(node0 != node1))
                self.assertTrue(np.any(cell0 != cell1))


if __name__ == "__main__":
    unittest.main()
