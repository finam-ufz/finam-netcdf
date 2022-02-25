"""
NetCDF writer components.
"""
from datetime import datetime, timedelta

from functools import partial
import numpy as np
import xarray as xr

from finam.core.interfaces import ComponentStatus
from finam.core.sdk import Input, ATimeComponent, AComponent, CallbackInput
from finam.data.grid import Grid
from numpy import datetime64

from . import Layer


class NetCdfTimedWriter(ATimeComponent):
    """
    NetCDF writer component that writes in predefined time intervals.

    Usage:

    .. code-block:: python

       file = "path/to/file.nc"
       writer = NetCdfTimedWriter(
            path=file,
            inputs={
                "LAI": Layer(var="lai", x="lon", y="lat"),
                "SM": Layer(var="soil_moisture", x="lon", y="lat"),
            },
            time_var="time",
            start=datetime(2000, 1, 1),
            step=timedelta(days=1),
       )
    """

    def __init__(
        self,
        path: str,
        inputs: dict[str, Layer],
        time_var: str,
        start: datetime,
        step: timedelta,
    ):
        """
        Constructs a NetCDF writer for regular/predefined time steps.

        :param path: path to the output NetCDF file
        :param inputs: dictionary of inputs. Keys are input names, values are Layer object
        :param time_var: name of the time coordinate/variable in the output dataset
        :param start: starting time (of type datetime)
        :param step: time step (of type timedelta)
        """
        super(NetCdfTimedWriter, self).__init__()

        if start is not None and not isinstance(start, datetime):
            raise ValueError("Start must be None or of type datetime")
        if step is not None and not isinstance(step, timedelta):
            raise ValueError("Step must be None or of type timedelta")

        self._path = path
        self._input_dict = inputs
        self._step = step
        self._time = start
        self.time_var = time_var
        self.data_arrays = {}

        self._inputs = {inp: Input() for inp in self._input_dict.keys()}

        self._status = ComponentStatus.CREATED

    def initialize(self):
        super().initialize()

        self._status = ComponentStatus.INITIALIZED

    def connect(self):
        super().connect()

        self._status = ComponentStatus.CONNECTED

    def validate(self):
        super().validate()

        variables, coords = extract_vars_dims(self.inputs, self._input_dict, self.time)

        coords = dict({self.time_var: np.ndarray(0, dtype="datetime64[ns]")}, **coords)

        self.data_arrays = {
            layer.var: xr.DataArray(
                np.ndarray(
                    (0, coords[layer.y].shape[0], coords[layer.x].shape[0]), dtype=dtype
                ),
                coords=[coords[self.time_var], coords[layer.y], coords[layer.x]],
                dims=[self.time_var, layer.y, layer.x],
            )
            for name, (layer, dtype) in variables.items()
        }

        self._status = ComponentStatus.VALIDATED

    def update(self):
        super().update()

        for name, inp in self.inputs.items():
            layer = self._input_dict[name]
            data = inp.pull_data(self.time)

            if not isinstance(data, Grid):
                raise ValueError(
                    "Only data of type `Grid` can be added to NetCDF files."
                )

            var = self.data_arrays[layer.var]

            new_var = xr.DataArray(
                np.expand_dims(data.reshape(data.spec.nrows, data.spec.ncols), axis=0),
                coords=[
                    [datetime64(self.time, "ns")],
                    var.coords[layer.y],
                    var.coords[layer.x],
                ],
                dims=[self.time_var, layer.y, layer.x],
            )

            self.data_arrays[layer.var] = xr.concat((var, new_var), dim=self.time_var)

        self._time += self._step
        self._status = ComponentStatus.UPDATED

    def finalize(self):
        super().finalize()

        dataset = xr.Dataset(data_vars=self.data_arrays)
        dataset.to_netcdf(self._path, unlimited_dims=[self.time_var])
        dataset.close()

        self._status = ComponentStatus.FINALIZED


class NetCdfPushWriter(AComponent):
    """
    NetCDF writer component that writes on push to its inputs.

    Usage:

    .. code-block:: python

       file = "path/to/file.nc"
       writer = NetCdfPushWriter(
            path=file,
            inputs={
                "LAI": Layer(var="lai", x="lon", y="lat"),
                "SM": Layer(var="soil_moisture", x="lon", y="lat"),
            },
            time_var="time"
       )

    Note that all data sources must have the same time step!
    """

    def __init__(
        self,
        path: str,
        inputs: dict[str, Layer],
        time_var: str,
    ):
        """
        Constructs a NetCDF writer for push-based writing.

        :param path: path to the output NetCDF file
        :param inputs: dictionary of inputs. Keys are input names, values are Layer object
        :param time_var: name of the time coordinate/variable in the output dataset
        """
        super(NetCdfPushWriter, self).__init__()

        self._path = path
        self._input_dict = inputs
        self.time_var = time_var
        self.data_arrays = {}

        self._inputs = {
            inp: CallbackInput(partial(self.data_changed, inp))
            for inp in self._input_dict.keys()
        }

        self.last_update = None

        self._status = ComponentStatus.CREATED

    def initialize(self):
        super().initialize()

        self._status = ComponentStatus.INITIALIZED

    def connect(self):
        super().connect()

        self._status = ComponentStatus.CONNECTED

    def validate(self):
        super().validate()

        variables, coords = extract_vars_dims(
            self.inputs, self._input_dict, self.last_update
        )

        coords = dict({self.time_var: np.ndarray(0, dtype="datetime64[ns]")}, **coords)

        self.data_arrays = {
            layer.var: xr.DataArray(
                np.ndarray(
                    (0, coords[layer.y].shape[0], coords[layer.x].shape[0]), dtype=dtype
                ),
                coords=[coords[self.time_var], coords[layer.y], coords[layer.x]],
                dims=[self.time_var, layer.y, layer.x],
            )
            for name, (layer, dtype) in variables.items()
        }

        for name, inp in self.inputs.items():
            layer = self._input_dict[name]
            data = inp.pull_data(self.last_update)

            if not isinstance(data, Grid):
                raise ValueError(
                    "Only data of type `Grid` can be added to NetCDF files."
                )

            var = self.data_arrays[layer.var]

            new_var = xr.DataArray(
                np.expand_dims(data.reshape(data.spec.nrows, data.spec.ncols), axis=0),
                coords=[
                    [datetime64(self.last_update, "ns")],
                    var.coords[layer.y],
                    var.coords[layer.x],
                ],
                dims=[self.time_var, layer.y, layer.x],
            )

            self.data_arrays[layer.var] = xr.concat((var, new_var), dim=self.time_var)

        self._status = ComponentStatus.VALIDATED

    def update(self):
        super().update()
        self._status = ComponentStatus.UPDATED

    def finalize(self):
        super().finalize()

        dataset = xr.Dataset(data_vars=self.data_arrays)
        dataset.to_netcdf(self._path, unlimited_dims=[self.time_var])
        dataset.close()
        dataset.close()

        self._status = ComponentStatus.FINALIZED

    def data_changed(self, name, caller, time):
        if not isinstance(time, datetime):
            raise ValueError("Time must be of type datetime")

        if self.status == ComponentStatus.INITIALIZED:
            self.last_update = time
            return

        if time != self.last_update:
            lengths = [a.shape[0] for a in self.data_arrays.values()]
            if lengths.count(lengths[0]) != len(lengths):
                raise ValueError("Incomplete dataset for time %s" % (self.last_update,))

        self.last_update = time

        layer = self._input_dict[name]
        data = caller.pull_data(self.last_update)

        if not isinstance(data, Grid):
            raise ValueError("Only data of type `Grid` can be added to NetCDF files.")

        var = self.data_arrays[layer.var]

        new_var = xr.DataArray(
            np.expand_dims(data.reshape(data.spec.nrows, data.spec.ncols), axis=0),
            coords=[
                [datetime64(self.last_update, "ns")],
                var.coords[layer.y],
                var.coords[layer.x],
            ],
            dims=[self.time_var, layer.y, layer.x],
        )

        self.data_arrays[layer.var] = xr.concat((var, new_var), dim=self.time_var)

        self.update()


def extract_vars_dims(inputs, layers, t0):
    variables = {}
    x_dims = {}
    y_dims = {}

    for name, inp in inputs.items():
        layer = layers[name]
        data = inp.pull_data(t0)

        if not isinstance(data, Grid):
            raise ValueError("Only data of type `Grid` can be added to NetCDF files.")

        if layer.var in variables:
            raise ValueError("Duplicate variable %s." % (layer.var,))

        variables[layer.var] = layer, data.dtype

        if layer.x in y_dims:
            raise ValueError(
                "Y dimension '%s' is already defined."
                "Can't be redefined as X dimension by input '%s'" % (layer.x, name)
            )
        if layer.y in x_dims:
            raise ValueError(
                "X dimension '%s' is already defined."
                "Can't be redefined as Y dimension by input '%s'" % (layer.y, name)
            )

        if layer.x in x_dims:
            spec = x_dims[layer.x]
            if (
                spec.xll != data.spec.xll
                or spec.cell_size != data.spec.cell_size
                or spec.ncols != data.spec.ncols
            ):
                raise ValueError(
                    "X dimension '%s' is already defined."
                    "Definition differs from data provided by input '%s'"
                    % (layer.x, name)
                )
        else:
            x_dims[layer.x] = data.spec

        if layer.y in y_dims:
            spec = y_dims[layer.y]
            if (
                spec.yll != data.spec.yll
                or spec.cell_size != data.spec.cell_size
                or spec.nrows != data.spec.nrows
            ):
                raise ValueError(
                    "Y dimension '%s' is already defined."
                    "Definition differs from data provided by input '%s'"
                    % (layer.y, name)
                )
        else:
            y_dims[layer.y] = data.spec

    x_dims = {
        name: np.arange(spec.ncols) * spec.cell_size + (spec.xll + 0.5 * spec.cell_size)
        for name, spec in x_dims.items()
    }
    y_dims = {
        name: np.flip(
            np.arange(spec.nrows) * spec.cell_size + (spec.yll + 0.5 * spec.cell_size)
        )
        for name, spec in y_dims.items()
    }
    coords = dict(x_dims, **y_dims)

    return variables, coords
