"""
NetCDF reader components.
"""

from datetime import datetime
import numpy as np
import xarray as xr

from finam.core.interfaces import ComponentStatus
from finam.core.sdk import AComponent, Output, ATimeComponent
from finam.data.grid import Grid, GridSpec
from . import Layer


class NetCdfInitReader(AComponent):
    """
    NetCDF reader component that reads a single 2D data array at startup.

    Usage:

    .. code-block:: python

       path = "tests/data/lai.nc"
       reader = NetCdfInitReader(
           path, {"LAI": Layer(var="lai", x="lon", y="lat", fixed={"time": 0})}
       )
    """

    def __init__(self, path: str, outputs: dict[str, Layer], time: datetime = None):
        """
        Constructs a NetCDF reader for reading a single data grid.

        :param path: path to NetCDF file
        :param outputs: dictionary of outputs. Keys are output names, values are Layer object
        :param time: starting time stamp. Optional. Default '1900-01-01'.
        """
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
            grid = extract_grid(self.dataset, pars, pars.fixed)
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

        self.dataset.close()

        self._status = ComponentStatus.FINALIZED


class NetCdfTimeReader(ATimeComponent):
    """
    NetCDF reader component that steps along a date/time coordinate dimension of a dataset.

    Usage:

    .. code-block:: python

       path = "tests/data/lai.nc"
       reader = NetCdfTimeReader(
           path, {"LAI": Layer(var="lai", x="lon", y="lat")}, time_var="time"
       )
    """

    def __init__(
        self, path: str, outputs: dict[str, Layer], time_var: str, time_callback=None
    ):
        """
        Constructs a NetCDF reader for reading time series of data grid.

        :param path: path to NetCDF file
        :param outputs: dictionary of outputs. Keys are output names, values are Layer object
        :param time_var: time coordinate variable of the dataset
        :param time_callback: an optional callback for time stepping and indexing:
                              (step, last_time, last_index) -> (time, index)
        """
        super(NetCdfTimeReader, self).__init__()

        self.path = path
        self.output_vars = outputs
        self.time_var = time_var
        self.time_callback = time_callback
        self.dataset = None
        self.times = None

        self.time_index = None
        self._time = None
        self.step = 0

        self._status = ComponentStatus.CREATED

    def initialize(self):
        super().initialize()

        self._outputs = {o: Output() for o in self.output_vars.keys()}

        self._status = ComponentStatus.INITIALIZED

    def connect(self):
        super().connect()

        self.dataset = xr.open_dataset(self.path)

        if self.time_callback is None:
            times = self.dataset.coords[self.time_var].dt
            self.times = [
                datetime.combine(d, t) for d, t in zip(times.date.data, times.time.data)
            ]

            self.time_index = 0
            self._time = self.times[self.time_index]
        else:
            self._time, self.time_index = self.time_callback(self.step, None, None)

        for name, pars in self.output_vars.items():
            grid = extract_grid(self.dataset, pars, {self.time_var: self.time_index})
            self._outputs[name].push_data(grid, self._time)

        self._status = ComponentStatus.CONNECTED

    def validate(self):
        super().validate()
        self._status = ComponentStatus.VALIDATED

    def update(self):
        super().update()

        self.step += 1

        if self.time_callback is None:
            self.time_index += 1
            if self.time_index >= len(self.times):
                self._status = ComponentStatus.FINISHED
                return
            self._time = self.times[self.time_index]
        else:
            self._time, self.time_index = self.time_callback(
                self.step, self._time, self.time_index
            )

        for name, pars in self.output_vars.items():
            grid = extract_grid(self.dataset, pars, {self.time_var: self.time_index})
            self._outputs[name].push_data(grid, self._time)

        self._status = ComponentStatus.UPDATED

    def finalize(self):
        super().finalize()
        self._status = ComponentStatus.FINALIZED


def extract_grid(dataset, layer, fixed=None):
    """Extracts a 2D data array from a dataset"""
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

    fx = layer.fixed if fixed is None else dict(layer.fixed, **fixed)
    extr = variable.isel(fx)

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
