"""
NetCDF writer components.
"""

from datetime import datetime, timedelta
from functools import partial

import finam as fm
from netCDF4 import Dataset, date2num

from .tools import create_nc_framework, create_variable_list


class NetCdfTimedWriter(fm.TimeComponent):
    """
    NetCDF writer component that writes in predefined time intervals.

    Usage:

    .. testcode:: constructor

       from datetime import timedelta
       from finam_netcdf import NetCdfTimedWriter

       file = "tests/data/out.nc"
       writer = NetCdfTimedWriter(file, ["lai", "soil_moist"], step=timedelta(days=1))

    .. testcode:: constructor
        :hide:

        writer.initialize()

    Parameters
    ----------

    path : str
        Path to the NetCDF file to read.
    inputs : list of str or Variable.
        List of inputs. Input is either defined by name or a :class:`Variable` instance.
    step : datetime.timedelta
        Time step
    time_var : str or None, optional
        Name of the time coordinate. To create a static output file, set this to None.
        By default: "time"
    global_attrs : dict, optional
        global attributes for the NetCDF file inputed by the user.
    """

    def __init__(
        self,
        path,
        inputs,
        step,
        time_var="time",
        global_attrs=None,
    ):
        super().__init__()

        if step is not None and not isinstance(step, timedelta):
            raise ValueError("Step must be None or of type timedelta")

        self._path = path
        self.variables = create_variable_list(inputs)
        for var in self.variables:
            if var.static is None:
                var.static = not bool(time_var)
            if var.slices:
                msg = f"NetCDF: writer got slices information for variable: f{var.name}"
                raise ValueError(msg)
        self._step = step
        self.time_var = time_var
        self.global_attrs = global_attrs or {}
        self.dataset = None
        self.timestamp_counter = 0
        self.status = fm.ComponentStatus.CREATED

        if not isinstance(self.global_attrs, dict):
            raise ValueError("inputed global attributes must be of type dict")

    def _next_time(self):
        return self.time + self._step

    def _initialize(self):
        for var in self.variables:
            grid = var.info_kwargs.get("grid", None)
            units = var.info_kwargs.get("units", None)
            self.inputs.add(
                name=var.io_name,
                time=self.time,
                grid=grid,
                units=units,
                static=var.static,
                **var.get_meta(),
            )

        self.dataset = Dataset(self._path, "w")

        self.create_connector(pull_data=[var.io_name for var in self.variables])

    def _connect(self, start_time):
        self.try_connect(start_time=start_time)
        if self.status != fm.ComponentStatus.CONNECTED:
            return

        self._time = start_time
        create_nc_framework(
            self.dataset,
            self.time_var,
            self._time,
            self._step,
            self.connector.in_infos,
            self.connector.in_data,
            self.variables,
            self.global_attrs,
        )

        # adding time and var data to the first timestamp
        for var in self.variables:
            data = self.connector.in_data[var.io_name].magnitude
            if var.static:
                self.dataset[var.name][...] = data
            else:
                self.dataset[var.name][self.timestamp_counter, ...] = data
        if self.time_var:
            current_date = date2num(self._time, self.dataset[self.time_var].units)
            self.dataset[self.time_var][self.timestamp_counter] = current_date

    def _validate(self):
        pass

    def _update(self):
        self._time += self._step
        self.timestamp_counter += 1

        if not self.time_var:
            return

        for var in self.variables:
            if var.static:
                continue
            data = self.inputs[var.io_name].pull_data(self._time).magnitude
            self.dataset[var.name][self.timestamp_counter, ...] = data

        current_date = date2num(self._time, self.dataset[self.time_var].units)
        self.dataset[self.time_var][self.timestamp_counter] = current_date

    def _finalize(self):
        self.dataset.close()


class NetCdfStaticWriter(fm.Component):
    """
    NetCDF writer component for static inputs.

    Usage:

    .. testcode:: constructor

       from finam_netcdf import NetCdfStaticWriter

       file = "tests/data/out.nc"
       writer = NetCdfTimedWriter(file, ["lai"])

    .. testcode:: constructor
        :hide:

        writer.initialize()

    Parameters
    ----------

    path : str
        Path to the NetCDF file to read.
    inputs : list of str or Variable.
        List of inputs. Input is either defined by name or a :class:`Variable` instance.
    global_attrs : dict, optional
        global attributes for the NetCDF file inputed by the user.
    """

    def __init__(
        self,
        path,
        inputs,
        global_attrs=None,
    ):
        super().__init__()

        self._path = path
        self.variables = create_variable_list(inputs)
        for var in self.variables:
            if var.static is None:
                var.static = True
            elif not var.static:
                msg = f"NetCDF: static writer got non static variable: f{var.name}"
                raise ValueError(msg)
            if var.slices:
                msg = f"NetCDF: writer got slices information for variable: f{var.name}"
                raise ValueError(msg)
        self.global_attrs = global_attrs or {}
        self.dataset = None
        self.status = fm.ComponentStatus.CREATED

        if not isinstance(self.global_attrs, dict):
            raise ValueError("inputed global attributes must be of type dict")

    def _initialize(self):
        for var in self.variables:
            grid = var.info_kwargs.get("grid", None)
            units = var.info_kwargs.get("units", None)
            self.inputs.add(
                name=var.io_name,
                time=None,
                grid=grid,
                units=units,
                static=var.static,
                **var.get_meta(),
            )

        self.dataset = Dataset(self._path, "w")

        self.create_connector(pull_data=[var.io_name for var in self.variables])

    def _connect(self, start_time):
        self.try_connect(start_time=start_time)
        if self.status != fm.ComponentStatus.CONNECTED:
            return

        create_nc_framework(
            self.dataset,
            None,
            None,
            None,
            self.connector.in_infos,
            self.connector.in_data,
            self.variables,
            self.global_attrs,
        )

        # adding time and var data to the first timestamp
        for var in self.variables:
            self.dataset[var.name][...] = self.connector.in_data[var.io_name].magnitude

    def _validate(self):
        pass

    def _update(self):
        pass

    def _finalize(self):
        self.dataset.close()


class NetCdfPushWriter(fm.Component):
    """
    NetCDF writer component that writes on push to its inputs.

    Usage:

    .. testcode:: constructor

       from finam_netcdf import NetCdfPushWriter

       file = "tests/data/out.nc"
       writer = NetCdfPushWriter(file, ["lai", "soil_moisture"])

    .. testcode:: constructor
        :hide:

        writer.initialize()

    Note that all data sources must have the same time step!

    Parameters
    ----------

    path : str
        Path to the NetCDF file to read.
    inputs : list of str or Variable.
        List of inputs. Input is either defined by name or a :class:`Variable` instance.
    time_var : str
        Name of the time coordinate.
    time_unit : str, optional
        time unit given as a string: days, hours, minutes or seconds.
    global_attrs : dict, optional
            global attributes for the NetCDF file inputed by the user.
    """

    def __init__(
        self,
        path,
        inputs,
        time_var="time",
        time_unit="seconds",
        global_attrs=None,
    ):
        super().__init__()

        self._path = path
        self.variables = create_variable_list(inputs)
        for var in self.variables:
            if var.static is None:
                var.static = False
            if var.static:
                msg = f"NetCDF: push writer got a static input: f{var.name}"
                raise ValueError(msg)
            if var.slices:
                msg = f"NetCDF: writer got slices information for variable: f{var.name}"
                raise ValueError(msg)
        self.time_var = time_var
        self.dataset = None
        self.timestamp_counter = 0
        self.time_unit = time_unit
        self.global_attrs = global_attrs or {}
        self.last_update = None

        if not isinstance(self.global_attrs, dict):
            raise ValueError("Given global attributes must be of type dict.")

        self.all_inputs = set(var.io_name for var in self.variables)
        self.pushed_inputs = set()

        self._status = fm.ComponentStatus.CREATED

    def _initialize(self):
        for var in self.variables:
            grid = var.info_kwargs.get("grid", None)
            units = var.info_kwargs.get("units", None)
            self.inputs.add(
                io=fm.CallbackInput(
                    name=var.io_name,
                    callback=partial(self._data_changed, var.io_name),
                    time=None,
                    grid=grid,
                    units=units,
                    **var.get_meta(),
                )
            )
        self.dataset = Dataset(self._path, "w")
        self.create_connector(pull_data=[var.io_name for var in self.variables])

    def _connect(self, start_time):
        self.try_connect(start_time)

        if self.status != fm.ComponentStatus.CONNECTED:
            return

        create_nc_framework(
            self.dataset,
            self.time_var,
            start_time,
            self.time_unit,
            self.connector.in_infos,
            self.connector.in_data,
            self.variables,
            self.global_attrs,
        )

        current_date = date2num(start_time, self.dataset[self.time_var].units)
        self.dataset[self.time_var][self.timestamp_counter] = current_date

        # adding time and var data to the first timestamp
        for var in self.variables:
            data = self.connector.in_data[var.io_name].magnitude
            self.dataset[var.name][self.timestamp_counter, ...] = data

        self.timestamp_counter += 1

    def _validate(self):
        pass

    def _update(self):
        pass

    def _finalize(self):
        self.dataset.close()

    # pylint: disable-next=unused-argument
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

        if time != self.last_update and self.pushed_inputs:
            raise ValueError("Data not pushed for all inputs")

        self.last_update = time
        self.pushed_inputs.add(name)

        if self.pushed_inputs != self.all_inputs:
            return

        current_time = date2num(time, self.dataset[self.time_var].units)
        self.dataset[self.time_var][self.timestamp_counter] = current_time

        for var in self.variables:
            data = self.inputs[var.io_name].pull_data(time).magnitude
            self.dataset[var.name][self.timestamp_counter, ...] = data

        self.timestamp_counter += 1

        self.pushed_inputs.clear()

        self.update()
