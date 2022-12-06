"""
NetCDF writer components.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from functools import partial

import finam as fm
import numpy as np

# pylint: disable-next=W0611
import pint

# pylint: disable-next=W0611
import pint_xarray
import xarray as xr

from .tools import Layer


class NetCdfTimedWriter(fm.TimeComponent):
    """
    NetCDF writer component that writes in predefined time intervals.

    Usage:

    .. testcode:: constructor

       from datetime import datetime, timedelta
       from finam_netcdf import Layer, NetCdfTimedWriter

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

    .. testcode:: constructor
        :hide:

        writer.initialize()

    Parameters
    ----------

    path : str
        Path to the NetCDF file to read.
    inputs : dict of str, Layer
        Dictionary of inputs. Keys are output names, values are :class:`.Layer` objects.
    time_var : str
        Name of the time coordinate.
    start : datetime.datetime
        Starting time
    step : datetime.timedelta
        Time step
    """

    def __init__(
        self,
        path: str,
        inputs: dict[str, Layer],
        time_var: str,
        start: datetime,
        step: timedelta,
    ):
        super().__init__()

        if start is not None and not isinstance(start, datetime):
            raise ValueError("Start must be None or of type datetime")
        if step is not None and not isinstance(step, timedelta):
            raise ValueError("Step must be None or of type timedelta")

        self._path = path
        self._input_dict = inputs
        self._input_names = {v.var: k for k, v in inputs.items()}
        self._step = step
        self._time = start
        self.time_var = time_var
        self.data_arrays = {}
        self.timestamps = []
        self.attrs = {}
        self.coords = None

        self.status = fm.ComponentStatus.CREATED

    @property
    def next_time(self):
        return self.time + self._step

    def _initialize(self):
        for inp in self._input_dict.keys():
            self.inputs.add(name=inp, time=self.time, grid=None, units=None)

        self.create_connector(pull_data=list(self._input_dict.keys()))

    def _connect(self, start_time):
        self.try_connect(start_time=start_time)

        if self.status != fm.ComponentStatus.CONNECTED:
            return

        _variables, self.coords = _extract_vars_dims(
            self.connector.in_infos, self.connector.in_data, self._input_dict
        )

        self.data_arrays = {}
        self.timestamps.append(self.time)
        for name, layer in self._input_dict.items():
            data = self.connector.in_data[name].magnitude
            attrs = self.inputs[name].info.meta.copy()
            if "units" in attrs:
                attrs["units"] = str(attrs["units"])
            self.attrs[layer.var] = attrs
            self.data_arrays[layer.var] = data

    def _validate(self):
        pass

    def _update(self):
        self._time += self._step

        self.timestamps.append(self.time)
        for name, inp in self.inputs.items():
            layer = self._input_dict[name]
            new_var = inp.pull_data(self.time).magnitude

            var = self.data_arrays[layer.var]

            self.data_arrays[layer.var] = np.concatenate((var, new_var), axis=0)

    def _finalize(self):
        self.coords.update(
            dict(time=[np.datetime64(t).astype(datetime) for t in self.timestamps])
        )
        dataset = xr.Dataset(
            data_vars={
                k: (
                    ["time"]
                    + list(self.inputs[self._input_names[k]].info.grid.axes_names),
                    v,
                    self.attrs[k],
                )
                for k, v in self.data_arrays.items()
            },
            coords=self.coords,
        )
        dims = list(reversed([c for c in dataset.coords if c != self.time_var]))

        dataset = dataset.transpose(self.time_var, *dims)

        dataset.to_netcdf(self._path, unlimited_dims=[self.time_var])
        dataset.close()


class NetCdfPushWriter(fm.Component):
    """
    NetCDF writer component that writes on push to its inputs.

    Usage:

    .. testcode:: constructor

       from finam_netcdf import Layer, NetCdfPushWriter

       file = "path/to/file.nc"
       writer = NetCdfPushWriter(
            path=file,
            inputs={
                "LAI": Layer(var="lai", xyz=("lon", "lat")),
                "SM": Layer(var="soil_moisture", xyz=("lon", "lat")),
            },
            time_var="time"
       )

    .. testcode:: constructor
        :hide:

        writer.initialize()

    Note that all data sources must have the same time step!

    Parameters
    ----------

    path : str
        Path to the NetCDF file to read.
    inputs : dict of str, Layer
        Dictionary of inputs. Keys are output names, values are :class:`.Layer` objects.
    time_var : str
        Name of the time coordinate.
    """

    def __init__(
        self,
        path: str,
        inputs: dict[str, Layer],
        time_var: str,
    ):
        super().__init__()

        self._path = path
        self._input_dict = inputs
        self._input_names = {v.var: k for k, v in inputs.items()}
        self.time_var = time_var
        self.data_arrays = {}
        self.attrs = {}
        self.coords = None

        self.last_update = None
        self.timestamps = []

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

    def _connect(self, start_time):
        self.try_connect(start_time)

        if self.status != fm.ComponentStatus.CONNECTED:
            return

        _variables, self.coords = _extract_vars_dims(
            self.connector.in_infos, self.connector.in_data, self._input_dict
        )

        self.data_arrays = {}
        for name, layer in self._input_dict.items():
            data = self.connector.in_data[name].magnitude
            attrs = self.inputs[name].info.meta.copy()
            if "units" in attrs:
                attrs["units"] = str(attrs["units"])
            self.attrs[layer.var] = attrs
            self.data_arrays[layer.var] = data

    def _validate(self):
        pass

    def _update(self):
        pass

    def _finalize(self):
        self.coords.update(
            dict(time=[np.datetime64(t).astype(datetime) for t in self.timestamps])
        )
        dataset = xr.Dataset(
            data_vars={
                k: (
                    ["time"]
                    + list(self.inputs[self._input_names[k]].info.grid.axes_names),
                    v,
                    self.attrs[k],
                )
                for k, v in self.data_arrays.items()
            },
            coords=self.coords,
        )
        dims = list(reversed([c for c in dataset.coords if c != self.time_var]))

        dataset = dataset.transpose(self.time_var, *dims)

        dataset.to_netcdf(self._path, unlimited_dims=[self.time_var])
        dataset.close()

    def _data_changed(self, name, caller, time):
        if self.status in (
            fm.ComponentStatus.CONNECTED,
            fm.ComponentStatus.CONNECTING,
            fm.ComponentStatus.CONNECTING_IDLE,
        ):
            self.last_update = time
            if len(self.timestamps) == 0:
                self.timestamps.append(time)

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

            self.timestamps.append(time)

        self.last_update = time

        layer = self._input_dict[name]
        data = caller.pull_data(self.last_update)
        new_var = data.magnitude

        var = self.data_arrays[layer.var]
        self.data_arrays[layer.var] = np.concatenate((var, new_var), axis=0)

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
