import os.path
import unittest
from datetime import datetime, timedelta
from os import path
from tempfile import TemporaryDirectory

import finam as fm
import numpy as np
from finam import Composition, Info, UniformGrid
from finam.components.generators import CallbackGenerator
from netCDF4 import Dataset

from finam_netcdf import NetCdfPushWriter, NetCdfStaticWriter, NetCdfTimedWriter


def generate_grid(grid):
    return np.reshape(
        np.random.random(grid.data_size), grid.data_shape, order=grid.order
    )


def _mesh_axes_attrs(dim):
    return [{"axis": "XYZ"[i], "units": "m"} for i in range(dim)]


def _make_mesh_grids(points, cells, cell_types):
    dim = points.shape[1]
    axes_attrs = _mesh_axes_attrs(dim)
    axes_names = ["x", "y", "z"][:dim]
    grid_cells = fm.UnstructuredGrid(
        points=points,
        cells=cells,
        cell_types=cell_types,
        data_location=fm.Location.CELLS,
        axes_attributes=axes_attrs,
        axes_names=axes_names,
    )
    grid_nodes = fm.UnstructuredGrid(
        points=points,
        cells=cells,
        cell_types=cell_types,
        data_location=fm.Location.POINTS,
        axes_attributes=axes_attrs,
        axes_names=axes_names,
    )
    return grid_nodes, grid_cells


class TestWriter(unittest.TestCase):
    def test_time_writer(self):
        grid = UniformGrid((10, 5), data_location="POINTS")

        with TemporaryDirectory() as tmp:
            file = path.join(tmp, "test.nc")

            source1 = CallbackGenerator(
                callbacks={
                    "Grid": (lambda t: generate_grid(grid), Info(None, grid, units="m"))
                },
                start=datetime(2000, 1, 1),
                step=timedelta(days=1),
            )
            source2 = CallbackGenerator(
                callbacks={
                    "Grid": (lambda t: generate_grid(grid), Info(None, grid, units="m"))
                },
                start=datetime(2000, 1, 1),
                step=timedelta(days=1),
            )

            # creating global attrs to the NetCDF output file - optional
            global_attrs = {
                "project_name": "test_time_writer",
                "original_source": "FINAM – Python model coupling framework",
                "creator_url": "https://finam.pages.ufz.de",
                "institution": "Helmholtz Centre for Environmental Research - UFZ (Helmholtz-Zentrum für Umweltforschung GmbH UFZ)",
                "description": "FINAM test: test_time_writer",
                "created_date": datetime.now().strftime("%d-%m-%Y"),
            }

            writer = NetCdfTimedWriter(
                path=file,
                inputs=["lai", "lai2"],
                step=timedelta(days=1),
                global_attrs=global_attrs,
            )

            composition = Composition([source1, source2, writer])

            source1.outputs["Grid"] >> writer.inputs["lai"]
            source2.outputs["Grid"] >> writer.inputs["lai2"]

            composition.run(end_time=datetime(2000, 1, 31))

            self.assertTrue(os.path.isfile(file))
            dataset = Dataset(file)

            lai = dataset["lai"]
            dims = list(lai.dimensions)

            self.assertEqual(dims, ["time", "x", "y"])
            self.assertEqual(lai.shape, (31, 10, 5))

            times = dataset["time"][:]
            self.assertEqual(times[0], 0.0)
            self.assertEqual(times[-1], 30.0)

            dataset.close()

    def test_writer_reverse_axes_uniform(self):
        self._test_writer_reverse_axes(False)

    def test_writer_reverse_axes_rectilinear(self):
        self._test_writer_reverse_axes(True)

    def _test_writer_reverse_axes(self, rectilinear):
        grid1 = UniformGrid((10, 5), data_location="POINTS", crs="EPSG:3035")
        grid2 = UniformGrid((10, 5), data_location="POINTS", crs="EPSG:4326")
        grid3 = UniformGrid((10, 5), data_location="POINTS")
        if rectilinear:
            grid1 = grid1.to_rectilinear()
            grid2 = grid2.to_rectilinear()
            grid3 = grid3.to_rectilinear()

        with TemporaryDirectory() as tmp:
            file = path.join(tmp, "test.nc")

            source1 = CallbackGenerator(
                callbacks={
                    "Grid": (
                        lambda t: generate_grid(grid1),
                        Info(None, grid1, units="m"),
                    )
                },
                start=datetime(2000, 1, 1),
                step=timedelta(days=1),
            )
            source2 = CallbackGenerator(
                callbacks={
                    "Grid": (
                        lambda t: generate_grid(grid2),
                        Info(None, grid2, units="m"),
                    )
                },
                start=datetime(2000, 1, 1),
                step=timedelta(days=1),
            )
            source3 = CallbackGenerator(
                callbacks={
                    "Grid": (
                        lambda t: generate_grid(grid3),
                        Info(None, grid3, units="m"),
                    )
                },
                start=datetime(2000, 1, 1),
                step=timedelta(days=1),
            )

            writer = NetCdfTimedWriter(
                path=file,
                inputs=["lai", "lai2", "lai3"],
                step=timedelta(days=1),
                force_axes_reversed=True,
            )

            composition = Composition([source1, source2, source3, writer])

            source1.outputs["Grid"] >> writer.inputs["lai"]
            source2.outputs["Grid"] >> writer.inputs["lai2"]
            source3.outputs["Grid"] >> writer.inputs["lai3"]

            composition.run(end_time=datetime(2000, 1, 31))

            self.assertTrue(os.path.isfile(file))
            dataset = Dataset(file)

            lai = dataset["lai"]
            dims = list(lai.dimensions)

            self.assertEqual(dims, ["time", "y", "x"])
            self.assertEqual(lai.shape, (31, 5, 10))

            lai2 = dataset["lai2"]
            lai3 = dataset["lai3"]
            self.assertEqual(lai.grid_mapping, "crs_0")
            self.assertEqual(lai2.grid_mapping, "crs_1")
            self.assertFalse(hasattr(lai3, "grid_mapping"))

            crs0 = dataset["crs_0"]
            self.assertEqual(crs0.epsg_code, "EPSG:3035")
            crs0 = dataset["crs_1"]
            self.assertEqual(crs0.epsg_code, "EPSG:4326")

            times = dataset["time"][:]
            self.assertEqual(times[0], 0.0)
            self.assertEqual(times[-1], 30.0)

            dataset.close()

    def test_static_writer(self):
        grid = UniformGrid((10, 5), data_location="POINTS")

        with TemporaryDirectory() as tmp:
            file = path.join(tmp, "test.nc")

            source = fm.components.StaticSimplexNoise(
                info=fm.Info(None, grid=grid, units="m"),
                frequency=0.05,
                octaves=3,
                persistence=0.5,
            )

            # creating global attrs to the NetCDF output file - optional
            global_attrs = {
                "project_name": "test_static_writer",
                "original_source": "FINAM – Python model coupling framework",
                "creator_url": "https://finam.pages.ufz.de",
                "institution": "Helmholtz Centre for Environmental Research - UFZ (Helmholtz-Zentrum für Umweltforschung GmbH UFZ)",
                "description": "FINAM test: test_static_writer",
                "created_date": datetime.now().strftime("%d-%m-%Y"),
            }

            writer = NetCdfStaticWriter(
                path=file,
                inputs=["height"],
                global_attrs=global_attrs,
            )

            composition = Composition([source, writer])

            source["Noise"] >> writer["height"]

            composition.run()

            self.assertTrue(os.path.isfile(file))
            dataset = Dataset(file)

            height = dataset["height"]
            dims = list(height.dimensions)

            self.assertEqual(dims, ["x", "y"])
            self.assertEqual(height.shape, (10, 5))

            dataset.close()

    def test_push_writer(self):
        grid = UniformGrid((10, 5), data_location="POINTS")

        with TemporaryDirectory() as tmp:
            file = path.join(
                tmp,
                "test.nc",
            )

            source1 = CallbackGenerator(
                callbacks={"Grid": (lambda t: generate_grid(grid), Info(None, grid))},
                start=datetime(2000, 1, 1),
                step=timedelta(days=1),
            )
            source2 = CallbackGenerator(
                callbacks={"Grid": (lambda t: generate_grid(grid), Info(None, grid))},
                start=datetime(2000, 1, 1),
                step=timedelta(days=1),
            )
            writer = NetCdfPushWriter(path=file, inputs=["lai", "lai2"])

            composition = Composition([source1, source2, writer])

            source1.outputs["Grid"] >> writer.inputs["lai"]
            source2.outputs["Grid"] >> writer.inputs["lai2"]

            composition.run(end_time=datetime(2000, 1, 31))

            self.assertTrue(os.path.isfile(file))

            dataset = Dataset(file)
            lai = dataset["lai"]

            self.assertEqual(lai.dimensions, ("time", "x", "y"))

            times = dataset["time"][:]
            self.assertEqual(times[0], 0.0)
            self.assertEqual(times[-1], 30.0 * 86400)

            dataset.close()

    def test_push_writer_fail(self):
        """
        Writer should fail if inputs have unequal time steps
        """
        grid = UniformGrid((10, 5), data_location="POINTS")

        with TemporaryDirectory() as tmp:
            file = path.join(tmp, "test.nc")

            source1 = CallbackGenerator(
                callbacks={"Grid": (lambda t: generate_grid(grid), Info(None, grid))},
                start=datetime(2000, 1, 1),
                step=timedelta(days=1),
            )
            source2 = CallbackGenerator(
                callbacks={"Grid": (lambda t: generate_grid(grid), Info(None, grid))},
                start=datetime(2000, 1, 1),
                step=timedelta(days=2),
            )
            writer = NetCdfPushWriter(path=file, inputs=["lai", "lai2"])

            composition = Composition([source1, source2, writer])

            source1.outputs["Grid"] >> writer.inputs["lai"]
            source2.outputs["Grid"] >> writer.inputs["lai2"]

            with self.assertRaises(ValueError):
                composition.run(end_time=datetime(2000, 1, 31))

    def test_static_writer_unstructured(self):
        cases = [
            (
                "edge",
                1,
                np.array([[0.0], [1.0], [2.0], [3.0]]),
                np.array([[0, 1], [1, 2], [2, 3]], dtype=int),
                np.full(3, fm.CellType.LINE, dtype=int),
            ),
            (
                "face",
                2,
                np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1]], dtype=float),
                np.array([[0, 1, 2, -1], [1, 4, 5, 3]], dtype=int),
                np.array([fm.CellType.TRI, fm.CellType.QUAD], dtype=int),
            ),
            (
                "volume",
                3,
                np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [2, 0, 0],
                        [3, 0, 0],
                        [3, 1, 0],
                        [2, 1, 0],
                        [2, 0, 1],
                        [3, 0, 1],
                        [3, 1, 1],
                        [2, 1, 1],
                    ],
                    dtype=float,
                ),
                np.array(
                    [
                        [0, 1, 2, 3, -1, -1, -1, -1],
                        [4, 5, 6, 7, 8, 9, 10, 11],
                    ],
                    dtype=int,
                ),
                np.array([fm.CellType.TETRA, fm.CellType.HEX], dtype=int),
            ),
        ]

        for kind, mesh_dim, points, cells, cell_types in cases:
            with self.subTest(kind=kind):
                grid_nodes, grid_cells = _make_mesh_grids(points, cells, cell_types)

                with TemporaryDirectory() as tmp:
                    file = path.join(tmp, f"unstructured_{kind}.nc")
                    source = fm.components.StaticCallbackGenerator(
                        callbacks={
                            "node": (
                                lambda g=grid_nodes: np.arange(g.data_size),
                                Info(None, grid_nodes, units="1"),
                            ),
                            "cell": (
                                lambda g=grid_cells: np.arange(g.data_size) + 10,
                                Info(None, grid_cells, units="1"),
                            ),
                        }
                    )
                    writer = NetCdfStaticWriter(path=file, inputs=["node", "cell"])
                    composition = Composition([source, writer])
                    source.outputs["node"] >> writer.inputs["node"]
                    source.outputs["cell"] >> writer.inputs["cell"]
                    composition.run()

                    dataset = Dataset(file)
                    mesh_vars = [
                        v
                        for v in dataset.variables
                        if getattr(dataset[v], "cf_role", None) == "mesh_topology"
                    ]
                    self.assertEqual(len(mesh_vars), 1)
                    mesh_name = mesh_vars[0]
                    mesh_var = dataset[mesh_name]
                    self.assertEqual(mesh_var.topology_dimension, mesh_dim)

                    node_coords = mesh_var.node_coordinates.split()
                    self.assertEqual(len(node_coords), points.shape[1])
                    node_dim = dataset[node_coords[0]].dimensions[0]
                    self.assertEqual(len(dataset.dimensions[node_dim]), points.shape[0])

                    node_var = dataset["node"]
                    self.assertEqual(node_var.mesh, mesh_name)
                    self.assertEqual(node_var.location, "node")
                    self.assertEqual(node_var.dimensions, (node_dim,))

                    cell_var = dataset["cell"]
                    self.assertEqual(cell_var.mesh, mesh_name)
                    self.assertEqual(cell_var.location, kind)

                    conn_attr = f"{kind}_node_connectivity"
                    self.assertTrue(hasattr(mesh_var, conn_attr))
                    conn_name = getattr(mesh_var, conn_attr)
                    conn_var = dataset[conn_name]
                    self.assertEqual(conn_var.cf_role, conn_attr)
                    self.assertEqual(conn_var.start_index, 0)
                    cell_dim = conn_var.dimensions[0]
                    self.assertEqual(cell_var.dimensions, (cell_dim,))

                    if kind in ("face", "volume"):
                        self.assertTrue(hasattr(conn_var, "_FillValue"))
                        self.assertEqual(conn_var._FillValue, -1)

                    if kind == "volume":
                        self.assertTrue(hasattr(mesh_var, "volume_shape_type"))
                        vol_name = mesh_var.volume_shape_type
                        vol_var = dataset[vol_name]
                        self.assertEqual(vol_var.cf_role, "volume_shape_type")

                    dataset.close()

    def test_static_writer_unstructured_points(self):
        points = np.array([[0.0], [1.0], [2.0]])
        grid = fm.UnstructuredPoints(points=points, axes_attributes=_mesh_axes_attrs(1))

        with TemporaryDirectory() as tmp:
            file = path.join(tmp, "unstructured_points.nc")
            source = fm.components.StaticCallbackGenerator(
                callbacks={
                    "node": (
                        lambda g=grid: np.arange(g.data_size),
                        Info(None, grid, units="1"),
                    )
                }
            )
            writer = NetCdfStaticWriter(path=file, inputs=["node"])
            composition = Composition([source, writer])
            source.outputs["node"] >> writer.inputs["node"]
            composition.run()

            dataset = Dataset(file)
            mesh_vars = [
                v
                for v in dataset.variables
                if getattr(dataset[v], "cf_role", None) == "mesh_topology"
            ]
            self.assertEqual(len(mesh_vars), 1)
            mesh_name = mesh_vars[0]
            mesh_var = dataset[mesh_name]
            self.assertEqual(mesh_var.topology_dimension, 0)
            self.assertFalse(hasattr(mesh_var, "edge_node_connectivity"))
            self.assertFalse(hasattr(mesh_var, "face_node_connectivity"))
            self.assertFalse(hasattr(mesh_var, "volume_node_connectivity"))

            node_coords = mesh_var.node_coordinates.split()
            node_dim = dataset[node_coords[0]].dimensions[0]
            self.assertEqual(len(dataset.dimensions[node_dim]), points.shape[0])

            node_var = dataset["node"]
            self.assertEqual(node_var.mesh, mesh_name)
            self.assertEqual(node_var.location, "node")
            self.assertEqual(node_var.dimensions, (node_dim,))

            dataset.close()

    def test_timed_writer_unstructured(self):
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        cells = np.array([[0, 1, 2, -1], [1, 3, 2, -1]], dtype=int)
        cell_types = np.array([fm.CellType.TRI, fm.CellType.TRI], dtype=int)
        _grid_nodes, grid_cells = _make_mesh_grids(points, cells, cell_types)

        with TemporaryDirectory() as tmp:
            file = path.join(tmp, "unstructured_timed.nc")
            source = CallbackGenerator(
                callbacks={
                    "cell": (
                        lambda t, g=grid_cells: np.arange(g.data_size) + t.day,
                        Info(None, grid_cells, units="1"),
                    ),
                },
                start=datetime(2000, 1, 1),
                step=timedelta(days=1),
            )
            writer = NetCdfTimedWriter(
                path=file,
                inputs=["cell"],
                step=timedelta(days=1),
            )
            composition = Composition([source, writer])
            source.outputs["cell"] >> writer.inputs["cell"]
            composition.run(end_time=datetime(2000, 1, 2))

            dataset = Dataset(file)
            self.assertIn("time", dataset.dimensions)
            self.assertEqual(len(dataset.dimensions["time"]), 2)
            self.assertEqual(dataset["cell"].dimensions[0], "time")
            dataset.close()


if __name__ == "__main__":
    unittest.main()
