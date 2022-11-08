"""
NetCDF reader components.
"""
from datetime import datetime

import finam as fm
import xarray as xr

from .tools import Layer, extract_grid


class NetCdfStaticReader(fm.Component):
    """
    NetCDF reader component that reads a single 2D data array at startup.

    Usage:

    .. code-block:: python

       path = "tests/data/lai.nc"
       reader = NetCdfInitReader(
           path, {"LAI": Layer(var="lai", xyz=("lon", "lat"), fixed={"time": 0})}
       )
    """

    def __init__(self, path: str, outputs: dict[str, Layer]):
        """
        Constructs a NetCDF reader for reading a single data grid.

        :param path: path to NetCDF file
        :param outputs: dictionary of outputs. Keys are output names, values are Layer object
        :param time: starting time stamp. Optional. Default '1900-01-01'.
        """
        super().__init__()
        self.path = path
        self.output_vars = outputs
        self.dataset = None
        self.data = None
        self._time = None
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
                if self._time is None:
                    self._time = info.time
                else:
                    if self._time != info.time:
                        raise ValueError(
                            "Can't work with NetCDF variables with different timestamps"
                        )

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

    .. code-block:: python

       path = "tests/data/lai.nc"
       reader = NetCdfTimeReader(
           path, {"LAI": Layer(var="lai", x="lon", y="lat")}, time_var="time"
       )
    """

    def __init__(
        self,
        path: str,
        outputs: dict[str, Layer],
        time_var: str,
        time_limits=None,
        time_callback=None,
    ):
        """
        Constructs a NetCDF reader for reading time series of data grid.

        :param path: path to NetCDF file
        :param outputs: dictionary of outputs. Keys are output names, values are Layer object
        :param time_var: time coordinate variable of the dataset
        :param time_limits: tuple of start and end datetime (both inclusive)
        :param time_callback: an optional callback for time stepping and indexing:
                              (step, last_time, last_index) -> (time, index)
        """
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
        for o in self.output_vars.keys():
            self.outputs.add(name=o)
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
                info, grid = extract_grid(
                    self.dataset,
                    pars,
                    {self.time_var: self.time_indices[self.time_index]},
                )
                grid.name = name
                if self.time_callback is not None:
                    grid = fm.data.get_data(grid)
                self.data[name] = (info, grid)

        self.try_connect(
            time=self._time,
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
            _info, grid = extract_grid(
                self.dataset, pars, {self.time_var: self.time_indices[self.time_index]}
            )
            grid.name = name
            if self.time_callback is not None:
                grid = fm.data.get_data(grid)
            self._outputs[name].push_data(grid, self._time)

    def _finalize(self):
        self.dataset.close()
