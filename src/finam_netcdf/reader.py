"""
NetCDF reader components.
"""
from __future__ import annotations

from datetime import datetime

import finam as fm
import xarray as xr

from .tools import Layer, extract_grid


class NetCdfStaticReader(fm.Component):
    """
    NetCDF reader component that reads a single 2D data array per output at startup.

    Usage:

    .. testcode:: constructor

       from finam_netcdf import Layer, NetCdfStaticReader

       path = "tests/data/lai.nc"
       reader = NetCdfStaticReader(
           path,
           {"LAI": Layer(var="lai", xyz=("lon", "lat"), fixed={"time": 0})},
       )

    .. testcode:: constructor
        :hide:

        reader.initialize()

    Parameters
    ----------

    path : str
        Path to the NetCDF file to read.
    outputs : dict of str, Layer
        Dictionary of outputs. Keys are output names, values are Layer object
    """

    def __init__(self, path: str, outputs: dict[str, Layer]):
        super().__init__()
        self.path = path
        self.output_vars = outputs

        for layer in self.output_vars.values():
            layer.static = True

        self.dataset = None
        self.data = None
        self.status = fm.ComponentStatus.CREATED

    def _initialize(self):
        for o in self.output_vars.keys():
            self.outputs.add(name=o, static=True)
        self.create_connector()

    def _connect(self):
        if self.dataset is None:
            self.dataset = xr.open_dataset(self.path)
            self.data = {}
            for name, pars in self.output_vars.items():
                info, grid = extract_grid(self.dataset, pars, pars.fixed)
                grid.name = name
                self.data[name] = (info, grid)

        self.try_connect(
            push_infos={name: value[0] for name, value in self.data.items()},
            push_data={name: value[1] for name, value in self.data.items()},
        )

        if self.status == fm.ComponentStatus.CONNECTED:
            del self.data
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

       from finam_netcdf import Layer, NetCdfReader

       path = "tests/data/lai.nc"
       reader = NetCdfReader(
           path,
           {"LAI": Layer(var="lai", xyz=("lon", "lat"))},
           time_var="time"
       )

    .. testcode:: constructor
        :hide:

        reader.initialize()

    Parameters
    ----------

    path : str
        Path to the NetCDF file to read.
    outputs : dict of str, Layer
        Dictionary of outputs. Keys are output names, values are Layer object.
    time_var : str
        Name of the time coordinate.
    time_limits : tuple (datetime.datetime, datetime.datetime), optional
        Tuple of start and end datetime (both inclusive)
    time_callback : callable, optional
        An optional callback for time stepping and indexing:
        (step, last_time, last_index) -> (time, index)
    """

    def __init__(
        self,
        path: str,
        outputs: dict[str, Layer],
        time_var: str,
        time_limits=None,
        time_callback=None,
    ):
        super().__init__()

        self.path = path
        self.output_vars = outputs
        self.time_var = time_var
        self.time_callback = time_callback
        self.time_limits = time_limits
        self.dataset = None
        self.data = None
        self.times = None

        self.time_index = None
        self.time_indices = None
        self._time = None
        self.step = 0

        self._status = fm.ComponentStatus.CREATED

    @property
    def next_time(self):
        return None

    def _initialize(self):
        for o, layer in self.output_vars.items():
            self.outputs.add(name=o, static=layer.static)
        self.create_connector()

    def _connect(self):
        if self.dataset is None:
            self.dataset = xr.open_dataset(self.path)
            self.data = {}
            times = self.dataset.coords[self.time_var].dt

            if self.time_limits is None:
                self.time_indices = list(range(times.date.data.shape[0]))
            else:
                self.time_indices = []
                mn = self.time_limits[0]
                mx = self.time_limits[1]
                for i, (d, t) in enumerate(zip(times.date.data, times.time.data)):
                    tt = datetime.combine(d, t)
                    if (mn is None or tt >= mn) and (mx is None or tt <= mx):
                        self.time_indices.append(i)

            self.times = [
                datetime.combine(d, t) for d, t in zip(times.date.data, times.time.data)
            ]

            for i in range(len(self.times) - 1):
                if self.times[i] >= self.times[i + 1]:
                    raise ValueError(
                        f"NetCDF reader requires time dimension '{self.time_var}' to be in ascending order."
                    )

            if self.time_callback is None:
                self.time_index = 0
                t = self.times[self.time_indices[self.time_index]]
                self._time = datetime.combine(t.date(), t.time())
            else:
                self._time, self.time_index = self.time_callback(self.step, None, None)

            for name, pars in self.output_vars.items():
                time_var = (
                    {}
                    if pars.static
                    else {self.time_var: self.time_indices[self.time_index]}
                )

                info, grid = extract_grid(
                    self.dataset,
                    pars,
                    time_var,
                )
                grid.name = name
                info.time = self._time
                if self.time_callback is not None:
                    grid = fm.data.get_data(grid)
                self.data[name] = (info, grid)

        self.try_connect(
            push_infos={name: value[0] for name, value in self.data.items()},
            push_data={name: value[1] for name, value in self.data.items()},
        )

        if self.status == fm.ComponentStatus.CONNECTED:
            del self.data

    def _validate(self):
        pass

    def _update(self):
        self.step += 1

        if self.time_callback is None:
            self.time_index += 1
            if self.time_index >= len(self.time_indices):
                self._status = fm.ComponentStatus.FINISHED
                return
            self._time = self.times[self.time_indices[self.time_index]]
        else:
            self._time, self.time_index = self.time_callback(
                self.step, self._time, self.time_index
            )
        for name, pars in self.output_vars.items():
            if pars.static:
                continue

            _info, grid = extract_grid(
                self.dataset, pars, {self.time_var: self.time_indices[self.time_index]}
            )
            grid.name = name
            if self.time_callback is not None:
                grid = fm.data.get_data(grid)
            self._outputs[name].push_data(grid, self._time)

    def _finalize(self):
        self.dataset.close()
