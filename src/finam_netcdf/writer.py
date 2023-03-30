"""
NetCDF writer components.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from functools import partial

import finam as fm
import numpy as np

from netCDF4 import Dataset, date2num

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
        self.dataset = None
        self.timestamp_counter = 0

        self.status = fm.ComponentStatus.CREATED

    @property
    def next_time(self):
        return self.time + self._step

    def _initialize(self):
        for inp in self._input_dict.keys():
            self.inputs.add(name=inp, time=self.time, grid=None, units=None)

        self.dataset = Dataset(self._path, "w")

        self.create_connector(pull_data=list(self._input_dict.keys()))

    def _connect(self, start_time):
        self.try_connect(start_time=start_time)
        if self.status != fm.ComponentStatus.CONNECTED:
            return

        _create_nc_framework(
            self.dataset,
            self.time_var,
            self._time,
            self._step,
            self.connector.in_infos,
            self.connector.in_data,
            self._input_dict,
        )

        # adding time and var data to the first timestamp
        for name in self._input_dict:
            self.dataset[name][self.timestamp_counter, :, :] = self.connector.in_data[
                name
            ].magnitude
            self.dataset[self.time_var][self.timestamp_counter] = date2num(
                self._time, self.dataset[self.time_var].units
            )

    def _validate(self):
        pass

    def _update(self):
        self._time += self._step
        self.timestamp_counter += 1
        for name, inp in self.inputs.items():
            self.dataset[name][self.timestamp_counter, :, :] = inp.pull_data(
                self._time
            ).magnitude
            self.dataset[self.time_var][self.timestamp_counter] = date2num(
                self._time, self.dataset[self.time_var].units
            )

    def _finalize(self):
        self.dataset.close()


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
        self.dataset = None
        self.timestamp_counter = 0

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
        self.dataset = Dataset(self._path, "w")

        self.create_connector(pull_data=list(self._input_dict.keys()))

    def _connect(self, start_time):
        self.try_connect(start_time)

        if self.status != fm.ComponentStatus.CONNECTED:
            return

        _create_nc_framework(
            self.dataset,
            self.time_var,
            self._time,
            self._step,
            self.connector.in_infos,
            self.connector.in_data,
            self._input_dict,
        )

        # adding time and var data to the first timestamp
        for name in self._input_dict:
            self.dataset[name][self.timestamp_counter, :, :] = self.connector.in_data[
                name
            ].magnitude
            self.dataset[self.time_var][self.timestamp_counter] = date2num(
                self._time, self.dataset[self.time_var].units
            )

    def _validate(self):
        pass

    def _update(self):
        self._time += self._step
        self.timestamp_counter += 1
        for name, inp in self.inputs.items():
            self.dataset[name][self.timestamp_counter, :, :] = inp.pull_data(
                self._time
            ).magnitude
            self.dataset[self.time_var][self.timestamp_counter] = date2num(
                self._time, self.dataset[self.time_var].units
            )

    def _finalize(self):
        self.dataset.close()

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

        self.dataset[layer.var][self.timestamp_counter, :, :] = data.magnitude
        self.dataset[self.time_var][self.timestamp_counter] = date2num(
            time, self.dataset[self.time_var].units
        )

        self.update()


def _create_nc_framework(
    dataset, time_var, start_date, time_freq, in_infos, in_data, layers
):
    """
    Creates coords, eg. (x, y), an empty time, and all other parameter variables, eg. temperature.
    """
    all_layers = []
    all_var = []
    equal_layers = True
    for name in in_data:
        layer = layers[name]
        if layer.var in all_var:
            raise ValueError(f"Duplicated variable {layer.var}.")
        else:
            all_layers.append(layer.xyz)
            all_var.append(layer.var)

    current_layer = all_layers[0]
    for lays in all_layers:
        if current_layer != lays:
            equal_layers = False
            break
    if not equal_layers:
        raise ValueError(
            f"NetCdfTimedWriter Inputs {all_var} have different layers: {all_layers}."
        )

    # creating time dim and var
    dim = dataset.createDimension(time_var, None)
    var = dataset.createVariable(time_var, np.float64, (time_var,))

    def days_hours_minutes(td):
        """funtion to get units of time in days, hours, minutes or seconds as str"""
        if td.days != 0:
            return "days"
        elif td.seconds // 3600 != 0:
            return "hours"
        elif (td.seconds // 60) % 60 != 0:
            return "minutes"
        else:
            return "seconds"

    freq = days_hours_minutes(time_freq)
    var.units = freq + " since " + str(start_date)
    var.calendar = "standard"  # standard may not be always the case

    just_once = True
    for name in in_data:
        # creating lat-lon var and dim just once in a while loop
        while just_once:
            grid_info = in_infos[name].grid
            for i, ax in enumerate(layer.xyz):
                if ax not in grid_info.axes_names:
                    raise ValueError(
                        f"Dimension {i} '{ax}' is not in the data for input {name}. "
                        f"Available axes are {grid_info.axes_names}"
                    )
                axis_values = in_infos[name].grid.data_axes[i]
                axis_type = in_infos[name].grid.data_axes[i].dtype
                dim = dataset.createDimension(ax, len(axis_values))
                var = dataset.createVariable(ax, axis_type, (ax,))
                dataset[ax][:] = axis_values
            just_once = False

        # creating var and dim other than time and coords X,Y
        dim = (time_var, current_layer[0], current_layer[1])
        var = dataset.createVariable(name, np.float64, dim)
        var.units = str(in_data[name].units)
