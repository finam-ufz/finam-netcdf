"""
NetCDF reader components.
"""
from __future__ import annotations

import finam as fm
from netCDF4 import Dataset

from .tools import (
    create_time_dim,
    extract_data,
    extract_info,
    extract_time,
    extract_variables,
)


class NetCdfStaticReader(fm.Component):
    """
    NetCDF reader component that reads a single 2D data array per output at startup.

    Usage:

    .. testcode:: constructor

       from finam_netcdf import Variable, NetCdfStaticReader

       path = "tests/data/lai.nc"

       # automatically determine data variables
       reader = NetCdfStaticReader(path)

       # explicit data variables
       reader = NetCdfStaticReader(path, [Variable("lai", slices={"time": 0})])

    .. testcode:: constructor
        :hide:

        reader.initialize()

    Parameters
    ----------

    path : str
        Path to the NetCDF file to read.
    outputs : list of Variable or str
        List of outputs. Output is either defined by name or a :class:`Variable` instance.
        By default all NetCDF variables found in the file.
    """

    def __init__(self, path, outputs=None):
        super().__init__()
        self.path = path
        self.variables = outputs
        self.dataset = None
        self._infos = None
        self._data = None
        self.status = fm.ComponentStatus.CREATED

    def _initialize(self):
        self.dataset = Dataset(self.path)
        self.variables = extract_variables(
            self.dataset, self.variables, only_static=True
        )
        for var in self.variables:
            self.outputs.add(name=var.io_name, static=True)
        self.create_connector()

    def _connect(self, start_time):
        if self._infos is None:
            self._data = {}
            self._infos = {}
            for var in self.variables:
                self._infos[var.io_name] = extract_info(self.dataset, var)
                self._data[var.io_name] = extract_data(self.dataset, var)

        self.try_connect(start_time, push_infos=self._infos, push_data=self._data)

        if self.status == fm.ComponentStatus.CONNECTED:
            del self._data
            del self._infos
            self.dataset.close()
            del self.dataset

    def _validate(self):
        pass

    def _update(self):
        pass

    def _finalize(self):
        pass


class NetCdfReader(fm.TimeComponent):
    """
    NetCDF reader component that steps along a date/time coordinate dimension of a dataset.

    Usage:

    .. testcode:: constructor

       from finam_netcdf import Variable, NetCdfReader

       path = "tests/data/lai.nc"

       # automatically determine data variables
       reader = NetCdfReader(path)

       # explicit data variables
       reader = NetCdfReader(path, outputs=["lai"])

       # explicit data variables with additional information
       reader = NetCdfReader(path, outputs=[Variable("lai", slices={"time": 0})])

    .. testcode:: constructor
        :hide:

        reader.initialize()

    Parameters
    ----------

    path : str
        Path to the NetCDF file to read.
    outputs : list of str or Variable, optional
        List of outputs. Output is either defined by name or a :class:`Variable` instance.
        By default all NetCDF variables found in the file.
    time_limits : tuple (datetime.datetime, datetime.datetime), optional
        Tuple of start and end datetime (both inclusive)
    time_callback : callable, optional
        An optional callback for time stepping and indexing:
        (step, last_time, last_index) -> (time, index)
    """

    def __init__(
        self,
        path,
        outputs=None,
        time_limits=None,
        time_callback=None,
    ):
        super().__init__()

        self.path = path
        self.variables = outputs
        self.time_var = None
        self.time_callback = time_callback
        self.time_limits = time_limits
        self.dataset = None
        self._init_data = {}
        self.output_infos = {}
        self.times = None
        self.time_index = None
        self.time_indices = None
        self.step = 0
        self.data_pushed = False

        self._status = fm.ComponentStatus.CREATED

    @property
    def next_time(self):
        return None

    def _initialize(self):
        self.dataset = Dataset(self.path)
        self.time_var = extract_time(self.dataset)
        self.variables = extract_variables(self.dataset, self.variables)
        for var in self.variables:
            self.outputs.add(name=var.io_name, static=var.static)

        self._process_initial_data()
        self.create_connector()

    def _connect(self, start_time):
        if self.data_pushed:
            self.try_connect(start_time)
        else:
            self.data_pushed = True
            self.try_connect(
                start_time,
                push_data=self._init_data,
                push_infos=self.output_infos,
            )

        if self.status == fm.ComponentStatus.CONNECTED:
            del self._init_data

    def _process_initial_data(self):
        if self.time_var is not None:
            self.times = create_time_dim(self.dataset, self.time_var)

            if self.time_limits is None:
                self.time_indices = list(range(len(self.times)))
            else:
                self.time_indices = []
                mn, mx = self.time_limits
                for index, time in enumerate(self.times):
                    if (mn is None or time >= mn) and (mx is None or time <= mx):
                        self.time_indices.append(index)

            for i in range(len(self.times) - 1):
                if self.times[i] >= self.times[i + 1]:
                    msg = f"NetCDF reader requires time dimension '{self.time_var}' to be in ascending order."
                    raise ValueError(msg)

            if self.time_callback is None:
                self.time_index = 0
                self._time = self.times[self.time_indices[self.time_index]]
            else:
                self._time, self.time_index = self.time_callback(self.step, None, None)
        else:
            self.time_indices, self.time_index = [0], 0

        for var in self.variables:
            info = extract_info(self.dataset, var, self._time)
            data = extract_data(
                self.dataset, var, self.time_var, self.time_indices[self.time_index]
            )
            self._init_data[var.io_name] = data
            self.output_infos[var.io_name] = info

    def _validate(self):
        pass

    def _update(self):
        self.step += 1

        if self.time_callback is None:
            self.time_index += 1
        else:
            self._time, self.time_index = self.time_callback(
                self.step, self._time, self.time_index
            )
        # this also catches the case for no time dimension
        if self.time_index >= len(self.time_indices):
            # for a "static reader" don't set status to finished
            if self.time_var is not None:
                self._status = fm.ComponentStatus.FINISHED
            return

        if self.time_callback is None:
            self._time = self.times[self.time_indices[self.time_index]]

        for var in self.variables:
            if var.static:
                continue

            data = fm.UNITS.Quantity(
                extract_data(
                    self.dataset, var, self.time_var, self.time_indices[self.time_index]
                ),
                self.output_infos[var.io_name].units,
            )
            self._outputs[var.io_name].push_data(data, self._time)

    def _finalize(self):
        self.dataset.close()
