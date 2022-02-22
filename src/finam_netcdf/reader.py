from datetime import datetime
import numpy as np
import xarray as xr

from finam.core.interfaces import ComponentStatus
from finam.core.sdk import ATimeComponent, Output

from finam.data.grid import Grid, GridSpec


class Layer:
    def __init__(self, var, x, y, fixed):
        self.var = var
        self.x = x
        self.y = y
        self.fixed = fixed


class NetCdfReader(ATimeComponent):

    def __init__(self, path, outputs):
        super(NetCdfReader, self).__init__()
        self.path = path
        self.output_vars = outputs
        self.dataset = None

    def initialize(self):
        super().initialize()

        self._outputs = {o: Output() for o in self.output_vars.keys()}

        self._status = ComponentStatus.INITIALIZED

    def connect(self):
        super().connect()

        self.dataset = xr.open_dataset(self.path)
        for name, pars in self.output_vars.items():
            grid = extract_grid(self.dataset, pars)
            self._outputs[name].push_data(grid, self._time)

        self._status = ComponentStatus.CONNECTED

    def validate(self):
        super().validate()
        self._status = ComponentStatus.VALIDATED

    def update(self):
        super().update()
        self._status = ComponentStatus.UPDATED

    def finalize(self):
        super().finalize()
        self._status = ComponentStatus.FINALIZED


def extract_grid(dataset, layer):
    variable = dataset[layer.var].load()
    x = variable.coords[layer.x]
    y = variable.coords[layer.y]

    xmin = x.data.min()
    xmax = x.data.max()
    ymin = y.data.min()
    ymax = y.data.max()

    cellsize_x = (xmax - xmin) / (x.shape[0] - 1)
    cellsize_y = (ymax - ymin) / (y.shape[0] - 1)

    if abs(cellsize_x - cellsize_y) > 1e-8:
        raise ValueError("Only raster data with equal resolution in x and y direction is supported.")

    grid: Grid = Grid(GridSpec(
        ncols=x.shape[0],
        nrows=y.shape[0],
        cell_size=cellsize_x, xll=xmin - 0.5 * cellsize_x, yll=ymin - 0.5 * cellsize_x))

    extr = variable.isel(layer.fixed)
    coords = {}
    for j, yy in enumerate(y.data):
        for i, xx in enumerate(x.data):
            ii, jj = grid.to_cell(xx, yy)
            coords[layer.x] = i
            coords[layer.y] = j
            val = extr.isel(coords)
            grid.set(ii, jj, val.data)

    return grid
