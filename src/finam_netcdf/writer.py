"""
NetCDF writer components.
"""
from datetime import datetime, timedelta
import xarray as xr

from finam.core.interfaces import ComponentStatus
from finam.core.sdk import Input, ATimeComponent
from finam.data.grid import Grid

from . import Layer


class NetCdfTimedWriter(ATimeComponent):
    def __init__(self, path: str, inputs: dict[str, Layer], start: datetime, step: timedelta):
        super(NetCdfTimedWriter, self).__init__()

        if start is not None and not isinstance(start, datetime):
            raise ValueError("Start must be None or of type datetime")
        if step is not None and not isinstance(step, timedelta):
            raise ValueError("Step must be None or of type timedelta")

        self._path = path
        self._input_dict = inputs
        self._step = step
        self._time = start
        self.dataset = None

        self._inputs = {inp: Input() for inp in self._input_dict.keys()}

        self._status = ComponentStatus.CREATED

    def initialize(self):
        super().initialize()

        self._status = ComponentStatus.INITIALIZED

    def connect(self):
        super().connect()

        self._status = ComponentStatus.CONNECTED

    def validate(self):
        super().validate()

        variables = set()
        x_dims = {}
        y_dims = {}

        for (name, inp), (_n, layer) in zip(self.inputs.items(), self._input_dict.items()):
            data = inp.pull_data(self.time)

            if not isinstance(data, Grid):
                raise ValueError("Only data of type `Grid` can be added to NetCDF files.")

            if layer.var in variables:
                raise ValueError("Duplicate variable %s." % (layer.var,))

            variables.add(layer.var)

            if layer.x in x_dims:
                spec = x_dims[layer.x]
                if spec.xll != data.spec.xll \
                        or spec.cell_size != data.spec.cell_size \
                        or spec.ncols != data.spec.ncols:
                    raise ValueError("X dimension '%s' is already defined."
                                     "Definition differs from data provided by input '%s'" % (layer.x, name))
            else:
                x_dims[layer.x] = data.spec

            if layer.y in y_dims:
                spec = y_dims[layer.y]
                if spec.yll != data.spec.yll \
                        or spec.cell_size != data.spec.cell_size \
                        or spec.nrows != data.spec.nrows:
                    raise ValueError("Y dimension '%s' is already defined."
                                     "Definition differs from data provided by input '%s'" % (layer.y, name))
            else:
                y_dims[layer.y] = data.spec

        self.dataset = xr.Dataset()

        self._status = ComponentStatus.VALIDATED

    def update(self):
        super().update()

        values = [self._inputs[inp].pull_data(self.time) for inp in self.inputs.keys()]

        self._time += self._step
        self._status = ComponentStatus.UPDATED

    def finalize(self):
        super().finalize()

        # TODO: save dataset

        self._status = ComponentStatus.FINALIZED
