from datetime import datetime
import numpy as np
import xarray as xr

from finam.core.interfaces import ComponentStatus
from finam.core.sdk import AComponent, Output

from finam.data.grid import Grid, GridSpec


class Layer:
    def __init__(self, var, x, y, fixed):
        self.var = var
        self.x = x
        self.y = y
        self.fixed = fixed


class NetCdfInitReader(AComponent):
    def __init__(self, path, outputs, time=None):
        super(NetCdfInitReader, self).__init__()
        if time is not None and not isinstance(time, datetime):
            raise ValueError("Time must be None or of type datetime")

        self.path = path
        self.output_vars = outputs
        self.dataset = None

        self._time = datetime(1900, 1, 1) if time is None else time

        self._status = ComponentStatus.CREATED

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
        raise ValueError(
            "Only raster data with equal resolution in x and y direction is supported."
        )

    extr = variable.isel(layer.fixed)

    if len(extr.dims) != 2:
        raise ValueError("NetCDF variable %s has dimensions != 2" % (layer.var,))

    if extr.dims[0] == layer.x and extr.dims[1] == layer.y:
        transpose = True
    elif extr.dims[0] == layer.y and extr.dims[1] == layer.x:
        transpose = False
    else:
        raise ValueError(
            "NetCDF variable %s dimensions do not include x and y (%s, %s)"
            % (layer.var, layer.x, layer.y)
        )

    x_flip = x.data[0] > x.data[-1]
    y_flip = y.data[0] < y.data[-1]

    arr = extr.data
    if transpose:
        arr = arr.T
    if y_flip:
        arr = np.flipud(arr)
    if x_flip:
        arr = np.fliplr(arr)

    flat = arr.flatten()

    grid: Grid = Grid(
        GridSpec(
            ncols=x.shape[0],
            nrows=y.shape[0],
            cell_size=cellsize_x,
            xll=xmin - 0.5 * cellsize_x,
            yll=ymin - 0.5 * cellsize_x,
        ),
        data=flat,
    )

    return grid
