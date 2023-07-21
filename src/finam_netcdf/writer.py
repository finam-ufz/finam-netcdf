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
    step : datetime.timedelta
        Time step
    global_attrs : dict, optional
            global attributes for the NetCDF file inputed by the user
    """

    def __init__(
        self,
        path: str,
        inputs: dict[str, Layer],
        time_var: str,
        step: timedelta,
        global_attrs={},
    ):
        super().__init__()

        if step is not None and not isinstance(step, timedelta):
            raise ValueError("Step must be None or of type timedelta")

        self._path = path
        self._input_dict = inputs
        self._input_names = {v.var: k for k, v in inputs.items()}
        self._step = step
        self.time_var = time_var
        self.global_attrs = global_attrs
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

        self._time = start_time
        _create_nc_framework(
            self.dataset,
            self.time_var,
            self._time,
            self._step,
            self.connector.in_infos,
            self.connector.in_data,
            self._input_dict,
            self.global_attrs,
        )

        # adding time and var data to the first timestamp
        for key_name in self._input_dict:
            var_name = self._input_dict[key_name].var
            data = self.connector.in_data[key_name].magnitude
            self.dataset[var_name][self.timestamp_counter, :, :] = data
            current_date = date2num(self._time, self.dataset[self.time_var].units)
            self.dataset[self.time_var][self.timestamp_counter] = current_date

    def _validate(self):
        pass

    def _update(self):
        self._time += self._step
        self.timestamp_counter += 1
        for key_name, inp in self.inputs.items():
            var_name = self._input_dict[key_name].var
            data = inp.pull_data(self._time).magnitude
            self.dataset[var_name][self.timestamp_counter, :, :] = data
            current_date = date2num(self._time, self.dataset[self.time_var].units)
            self.dataset[self.time_var][self.timestamp_counter] = current_date

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
            time_var="time",
            step=timedelta(days=1),
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
    step : datetime.timedelta
        Time step
    global_attrs : dict, optional
            global attributes for the NetCDF file inputed by the user
    """

    def __init__(
        self,
        path: str,
        inputs: dict[str, Layer],
        time_var: str,
        step: timedelta,
        global_attrs={},
    ):
        super().__init__()

        self._path = path
        self._input_dict = inputs
        self._input_names = {v.var: k for k, v in inputs.items()}
        self.time_var = time_var
        self.dataset = None
        self.timestamp_counter = 0
        self._step = step
        self.global_attrs = global_attrs
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
        self.dataset = Dataset(self._path, "w")

        self.create_connector(pull_data=list(self._input_dict.keys()))

    def _connect(self, start_time):
        self.try_connect(start_time)

        if self.status != fm.ComponentStatus.CONNECTED:
            return

        _create_nc_framework(
            self.dataset,
            self.time_var,
            start_time,
            self._step,
            self.connector.in_infos,
            self.connector.in_data,
            self._input_dict,
            self.global_attrs,
        )

        # adding time and var data to the first timestamp
        # TODO: self.timestamp_counter is increase to 1 in _data_changed & therefore the -1 here
        # it seems that removing that statement gives the wrong time step from1 to 30, instead of 0 to 30.
        for key_name in self._input_dict:
            var_name = self._input_dict[key_name].var
            data = self.connector.in_data[key_name].magnitude
            self.dataset[var_name][self.timestamp_counter - 1, :, :] = data
            current_date = date2num(start_time, self.dataset[self.time_var].units)
            self.dataset[self.time_var][self.timestamp_counter - 1] = current_date

    def _validate(self):
        pass

    def _update(self):
        pass

    def _finalize(self):
        self.dataset.close()

    def _data_changed(self, name, caller, time):
        if self.status in (
            fm.ComponentStatus.CONNECTED,
            fm.ComponentStatus.CONNECTING,
            fm.ComponentStatus.CONNECTING_IDLE,
        ):
            self.last_update = time
            if self.timestamp_counter == 0:
                self.timestamp_counter += 1
            return

        if not isinstance(time, datetime):
            raise ValueError("Time must be of type datetime")

        if self.status == fm.ComponentStatus.INITIALIZED:
            self.last_update = time
            return

        layer = self._input_dict[name]

        if time != self.last_update:
            data = caller.pull_data(self.last_update).magnitude
            self.dataset[layer.var][self.timestamp_counter, :, :] = data
            current_time = date2num(time, self.dataset[self.time_var].units)
            self.dataset[self.time_var][self.timestamp_counter] = current_time
            self.timestamp_counter += 1

        self.last_update = time

        self.update()


def _create_nc_framework(
    dataset,
    time_var,
    start_date,
    time_step_units,
    in_infos,
    in_data,
    layers,
    global_attrs={},
):
    """
    creates a NetCDF with XYZ coords data, and empties time dimension and parameter variables.

        Parameters
        ----------
        dataset : netCDF4._netCDF4.Dataset
            empty NetCDF file
        time_var : str
            name of the time variable
        start_date : datetime.datetime
            starting time
        time_freq : datetime.datetime
            time stepping
        in_infos : dict
            grid data and units for each output variable
        in_data : dict
            array data and units for each output variable
        layers : list
            Layer information for each variable:
            Layer(var=--, xyz=(--, --, --), fixed={--}, static=--))
        global_attrs : dict, optional
            global attributes for the NetCDF file inputed by the user

        Raises
        ------
        ValueError
            If there is a duplicated output parameter varaible.
        ValueError
            If the names of the XYZ coordinates do not match for all variables.
    """
    coordinates = []
    variables = []
    for parameter in in_data:
        layer = layers[parameter]
        if layer.var in variables:
            raise ValueError(f"Duplicated variable {layer.var}.")
        else:
            coordinates.append(layer.xyz)
            variables.append(layer.var)

    equal_layers = True
    for l in coordinates:
        if coordinates[0] != l:
            equal_layers = False
            break
    if not equal_layers:
        raise ValueError(
            f"NetCdfTimedWriter Inputs {variables} have different layers: {coordinates}."
        )

    # adding general attributes
    dataset.setncatts(global_attrs)

    # creating time dim and var
    dataset.createDimension(time_var, None)
    t_var = dataset.createVariable(time_var, np.float64, (time_var,))

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

    freq = days_hours_minutes(time_step_units)
    t_var.units = freq + " since " + str(start_date)
    t_var.calendar = "standard"

    # creating xyz dim and var | all var have the same ordered layer.xyz & data coords
    name = next(iter(in_infos))  # gets the first key output parameter name
    grid_info = in_infos[name].grid  # same for all outputs
    for i, ax in enumerate(layer.xyz):
        if ax not in grid_info.axes_names:
            raise ValueError(
                f"Dimension {i} '{ax}' is not in the data for input {name}. "
                f"Available axes are {grid_info.axes_names}"
            )
        axis_values = grid_info.data_axes[i]
        axis_type = grid_info.data_axes[i].dtype
        dataset.createDimension(ax, len(axis_values))
        dataset.createVariable(ax, axis_type, (ax,))
        dataset[ax][:] = axis_values
        dataset[ax].setncattr("axis", "XYZ"[i])

    # creating parameter variables
    for parameter in in_data:
        var_name = layers[parameter].var
        dim = (time_var,) + coordinates[0]  # time plus existing coords
        var = dataset.createVariable(var_name, np.float64, dim)
        var.units = str(in_data[parameter].units)
