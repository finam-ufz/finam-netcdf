"""
NetCDF reader components.
"""
from datetime import datetime

import xarray as xr
from finam import ComponentStatus, AComponent, ATimeComponent, Output

from . import Layer, extract_grid


class NetCdfInitReader(AComponent):
    """
    NetCDF reader component that reads a single 2D data array at startup.

    Usage:

    .. code-block:: python

       path = "tests/data/lai.nc"
       reader = NetCdfInitReader(
           path, {"LAI": Layer(var="lai", xyz=("lon", "lat"), fixed={"time": 0})}
       )
    """

    def __init__(self, path: str, outputs: dict[str, Layer], time: datetime = None):
        """
        Constructs a NetCDF reader for reading a single data grid.

        :param path: path to NetCDF file
        :param outputs: dictionary of outputs. Keys are output names, values are Layer object
        :param time: starting time stamp. Optional. Default '1900-01-01'.
        """
        super().__init__()
        if time is not None and not isinstance(time, datetime):
            raise ValueError("Time must be None or of type datetime")

        self.path = path
        self.output_vars = outputs
        self.dataset = None
        self.data = None
        self._time = datetime(1900, 1, 1) if time is None else time
        self._status = ComponentStatus.CREATED

    def _initialize(self):
        for o in self.output_vars.keys():
            self.outputs.add(name=o)
        self.create_connector()

    def _connect(self):
        if self.dataset is None:
            self.dataset = xr.open_dataset(self.path)
            self.data = {}
            for name, pars in self.output_vars.items():
                info, grid = extract_grid(self.dataset, pars, pars.fixed)
                grid.name = name
                print(grid)
                self.data[name] = (info, grid)

        self.try_connect(
            time=self._time,
            push_infos={name: value[0] for name, value in self.data.items()},
            push_data={name: value[1] for name, value in self.data.items()}
        )

        if self.status == ComponentStatus.CONNECTED:
            del self.data
            self.dataset.close()
            del self.dataset

    def _validate(self):
        pass

    def _update(self):
        pass

    def _finalize(self):
        pass


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
        self.times = None

        self.time_index = None
        self.time_indices = None
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
            self._time = self.times[self.time_indices[self.time_index]]
        else:
            self._time, self.time_index = self.time_callback(self.step, None, None)

        for name, pars in self.output_vars.items():
            grid = extract_grid(
                self.dataset, pars, {self.time_var: self.time_indices[self.time_index]}
            )
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
            if self.time_index >= len(self.time_indices):
                self._status = ComponentStatus.FINISHED
                return
            self._time = self.times[self.time_indices[self.time_index]]
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
