"""
NetCDF reader components.
"""
from __future__ import annotations

import finam as fm
from netCDF4 import Dataset

from .tools import Layer, create_time_dim, extract_grid, extract_layers


class NetCdfStaticReader(fm.Component):
    """
    NetCDF reader component that reads a single 2D data array per output at startup.

    Usage:

    .. testcode:: constructor

       from finam_netcdf import Layer, NetCdfStaticReader

       path = "tests/data/lai.nc"

       # automatically determine data variables
       reader = NetCdfStaticReader(path)

       # explicit data variables
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
        Dictionary of outputs. Keys are output names, values are :class:`.Layer` objects.
        If not given, the reader tries to determine all variables from the dataset.
    """

    def __init__(self, path: str, outputs: dict[str, Layer] = None):
        super().__init__()
        self.path = path
        self.output_layers = outputs
        self.dataset = None
        self.data = None
        self.status = fm.ComponentStatus.CREATED

    def _initialize(self):
        self.dataset = Dataset(self.path)

        if self.output_layers is None:
            _time_var, layers = extract_layers(self.dataset)
            self.output_layers = {}
            for l in layers:
                if l.static:
                    self.output_layers[l.var] = l
                else:
                    self.logger.warning(
                        "Skipping variable %s, as it is not static.", l.var
                    )
        else:
            for layer in self.output_layers.values():
                layer.static = True

        for o in self.output_layers.keys():
            self.outputs.add(name=o, static=True)

        self.create_connector()

    def _connect(self, start_time):
        if self.data is None:
            self.data = {}
            for name, layer in self.output_layers.items():
                info, data = extract_grid(self.dataset, layer, layer.fixed)
                self.data[name] = (info, data)

        self.try_connect(
            start_time,
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

       # automatically determine data variables
       reader = NetCdfReader(path)

       # explicit data variables
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
    outputs : dict of (str, Layer), optional
        Dictionary of outputs. Keys are output names, values are :class:`.Layer` objects.
        If not given, the reader tries to determine all variables from the dataset.
    time_var : str, optional
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
        outputs: dict[str, Layer] = None,
        time_var: str = None,
        time_limits=None,
        time_callback=None,
    ):
        super().__init__()

        self.path = path
        self.output_layers = outputs
        self.time_var = time_var

        if (self.output_layers is None) != (self.time_var is None):
            raise ValueError(
                "Only none or both of `outputs` and `time_var` must be None"
            )

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
        if self.output_layers is None:
            self.time_var, layers = extract_layers(self.dataset)
            self.output_layers = {l.var: l for l in layers}

        for o, layer in self.output_layers.items():
            self.outputs.add(name=o, static=layer.static)

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
                raise ValueError(
                    f"NetCDF reader requires time dimension '{self.time_var}' to be in ascending order."
                )

        if self.time_callback is None:
            self.time_index = 0
            self._time = self.times[self.time_indices[self.time_index]]
        else:
            self._time, self.time_index = self.time_callback(self.step, None, None)

        for name, layer in self.output_layers.items():
            time_index = None if layer.static else self.time_index
            info, data = extract_grid(
                self.dataset,
                layer,
                time_index,
                self.time_var,
                self._time,
            )
            info.time = self._time
            if self.time_callback is not None:
                data = fm.data.strip_time(data, info.grid)
            self._init_data[name] = data
            self.output_infos[name] = info

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
        for out in self.output_layers:
            name = self.output_layers[out].var

            if self.output_layers[out].static:
                continue

            data = self.dataset[name][self.time_indices[self.time_index], ...]
            data = fm.UNITS.Quantity(data, self.output_infos[out].units)

            if self.time_callback is not None:
                data = fm.data.strip_time(data, self.output_infos[out].grid)

            self._outputs[out].push_data(data, self._time)

    def _finalize(self):
        self.dataset.close()
