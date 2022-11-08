"""
NetCDF writer components.
"""
from datetime import datetime, timedelta
from functools import partial

import finam as fm
import numpy as np
import xarray as xr

from .tools import Layer


class NetCdfTimedWriter(fm.TimeComponent):
    """
    NetCDF writer component that writes in predefined time intervals.

    Usage:

    .. code-block:: python

       file = "path/to/file.nc"
       writer = NetCdfTimedWriter(
            path=file,
            inputs={
                "LAI": Layer(var="lai", xyz=("lon", "lat")),
                "SM": Layer(var="soil_moisture", xyz=("lon", "lat")),
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
        super().__init__()

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

        self.status = fm.ComponentStatus.CREATED

    @property
    def next_time(self):
        return self.time + self._step

    def _initialize(self):
        for inp in self._input_dict.keys():
            self.inputs.add(name=inp, time=self.time, grid=None, units=None)

        self.create_connector(pull_data=list(self._input_dict.keys()))

    def _connect(self):
        self.try_connect(time=self._time)

        if self.status != fm.ComponentStatus.CONNECTED:
            return

        _variables, coords = _extract_vars_dims(
            self.connector.in_infos, self.connector.in_data, self._input_dict
        )

        self.data_arrays = {
            layer.var: self.connector.in_data[name].assign_coords(coords)
            for name, layer in self._input_dict.items()
        }

    def _validate(self):
        pass

    def _update(self):
        self._time += self._step

        for name, inp in self.inputs.items():
            layer = self._input_dict[name]
            new_var = inp.pull_data(self.time)

            var = self.data_arrays[layer.var]

            self.data_arrays[layer.var] = xr.concat((var, new_var), dim=self.time_var)

    def _finalize(self):
        dataset = xr.Dataset(data_vars=self.data_arrays)
        dataset.to_netcdf(self._path, unlimited_dims=[self.time_var])


class NetCdfPushWriter(fm.Component):
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
        super().__init__()

        self._path = path
        self._input_dict = inputs
        self.time_var = time_var
        self.data_arrays = {}

        self.last_update = None

        self._status = fm.ComponentStatus.CREATED

    def _initialize(self):
        for inp in self._input_dict.keys():
            self.inputs.add(
                io=fm.CallbackInput(
                    name=inp,
                    callback=partial(self._data_changed, inp),
                    time=None,
                    grid=None,
                    units=None,
                )
            )

        self.create_connector(pull_data=list(self._input_dict.keys()))

    def _connect(self):
        self.try_connect()

        if self.status != fm.ComponentStatus.CONNECTED:
            return

        _variables, coords = _extract_vars_dims(
            self.connector.in_infos, self.connector.in_data, self._input_dict
        )

        self.data_arrays = {
            layer.var: self.connector.in_data[name].assign_coords(coords)
            for name, layer in self._input_dict.items()
        }

    def _validate(self):
        pass

    def _update(self):
        pass

    def _finalize(self):
        dataset = xr.Dataset(data_vars=self.data_arrays)
        dataset.to_netcdf(self._path, unlimited_dims=[self.time_var])
        dataset.close()

    def _data_changed(self, name, caller, time):
        if self.status in (
            fm.ComponentStatus.CONNECTED,
            fm.ComponentStatus.CONNECTING,
            fm.ComponentStatus.CONNECTING_IDLE,
        ):
            self.last_update = time
            return

        if not isinstance(time, datetime):
            raise ValueError("Time must be of type datetime")

        if self.status == fm.ComponentStatus.INITIALIZED:
            self.last_update = time
            return
        if time != self.last_update:
            lengths = [a.shape[0] for a in self.data_arrays.values()]
            if lengths.count(lengths[0]) != len(lengths):
                raise ValueError(f"Incomplete dataset for time {self.last_update}")

        self.last_update = time

        layer = self._input_dict[name]
        new_var = caller.pull_data(self.last_update)

        var = self.data_arrays[layer.var]
        self.data_arrays[layer.var] = xr.concat((var, new_var), dim=self.time_var)

        self.update()


def _extract_vars_dims(in_infos, in_data, layers):
    variables = {}
    max_dims = 3
    dims = [{}, {}, {}]

    for name, data in in_data.items():
        layer = layers[name]

        if layer.var in variables:
            raise ValueError(f"Duplicate variable {layer.var}.")

        grid_info = in_infos[name].grid
        variables[layer.var] = layer, data.dtype

        for i, ax in enumerate(layer.xyz):
            if ax not in grid_info.axes_names:
                raise ValueError(
                    f"Dimension {i} '{ax}' is not in the data for input {name}. "
                    f"Available axes are {grid_info.axes_names}"
                )
            for j in range(max_dims):
                if i != j and ax in dims[j]:
                    raise ValueError(
                        f"Dimension {i} '{ax}' is already defined. "
                        f"Definition differs from data provided by input {name}"
                    )
            curr_dim = dims[i]
            axis_values = in_infos[name].grid.data_axes[i]
            if ax in curr_dim:
                axis = curr_dim[ax]
                if not np.allclose(axis_values, axis):
                    raise ValueError(
                        f"Dimension {i} '{ax}' is already defined. "
                        f"Definition differs from data provided by input {name}"
                    )
            else:
                curr_dim[ax] = axis_values

    return variables, dict(dict(**dims[0], **dims[1], **dims[2]))
