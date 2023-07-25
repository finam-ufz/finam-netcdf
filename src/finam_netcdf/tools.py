"""NetCDF helper classes and functions"""
import finam as fm
import numpy as np
from netCDF4 import num2date

from .info_checker import DatasetInfo, check_order_reversed


class Layer:
    """
    Defines a NetCDF layer.

    Parameters
    ----------

    var : str
        Layer variable
    xyz : tuple of str
        Coordinate variables in xyz order
    fixed : dict of str, int
        Dictionary for further, fixed index coordinate variables (e.g. 'time')
    static : bool, optional
        Marks this layer/outputs as static. Defaults to ``False``.
    """

    def __init__(self, var: str, xyz=("x", "y"), fixed=None, static=False):
        self.var = var
        self.xyz = xyz
        self.fixed = fixed or {}
        self.static = static

    def __repr__(self):
        return f"Layer(var={self.var}, xyz={self.xyz}, fixed={self.fixed}, static={self.static})"


def extract_layers(dataset):
    """
    It extracts the layer information from a dataset following CF convention.


    Parameters
    ----------
    dataset : str
        NetCDF file

    Returns
    -------
    tuple
        tuple including time and Z variable names and layers information.
    """

    layers = []
    data_info = DatasetInfo(dataset)
    variables = data_info.data_dims_map
    time_var = data_info.time
    time_var = None if not time_var else time_var.pop()

    # extracting names of coordinates XYZ
    for var, dims in variables.items():
        static = var in data_info.static_data
        xyz = tuple(value for value in dims if value != time_var)
        order = data_info.get_axes_order(xyz)
        axes_reversed = check_order_reversed(order)
        if axes_reversed:
            xyz = xyz[::-1]

        layers.append(Layer(var, xyz, static=static))

    return time_var, layers


def extract_grid(dataset, layer, time_index=None, time_var=None, current_time=None):
    """Extracts a 2D data array from a dataset

    Parameters
    ----------
    dataset : netCDF4.DataSet
        The input dataset
    layer : Layer
        The layer definition
    time_index : int, optional
        index of current time
    time_var: str, optional
        Time variable string
    current_time: datetime.datetime
        (YYYY, M, D, H, S)
    """

    data_info = DatasetInfo(dataset)
    data_var = dataset[layer.var]

    # storing attributes of data_var in meta dict
    meta = {name: getattr(data_var, name) for name in data_var.ncattrs()}

    # gets the data for each time step as np.array if time is not None
    if isinstance(time_index, int):
        data_var = np.array(data_var[time_index, ...].filled(np.nan))

    # checks if axes were reversed or not
    order = data_info.get_axes_order(data_info.data_dims_map[layer.var])
    axes_reversed = check_order_reversed(order)

    # getting coordinates data
    axes = [np.asarray(dataset.variables[ax][:]).copy() for ax in layer.xyz]
    axes_attrs = [dataset.variables[ax].ncattrs() for ax in layer.xyz]

    # note: we use point-associated data here.
    grid = fm.RectilinearGrid(
        axes=[create_point_axis(ax) for ax in axes],
        axes_names=layer.xyz,
        data_location=fm.Location.CELLS,
        axes_reversed=axes_reversed,
        axes_attributes=axes_attrs,
    )

    # getting current time
    times = None
    if not layer.static and time_var in dataset.dimensions:
        times = current_time

    info = fm.Info(time=times, grid=grid, meta=meta)

    return info, fm.UNITS.Quantity(np.array(data_var[:]), info.units)


def create_point_axis(cell_axis):
    """Create a point axis from a cell axis"""
    diffs = np.diff(cell_axis)
    mid = cell_axis[:-1] + diffs / 2
    first = cell_axis[0] - diffs[0] / 2
    last = cell_axis[-1] + diffs[-1] / 2
    return np.concatenate(([first], mid, [last]))


def create_time_dim(dataset, time_var):
    """returns a list of datetime.datetime objects for a given NetCDF4 time varaible"""
    if "units" and "calendar" not in dataset[time_var].ncattrs():
        raise AttributeError(
            f"Variable {time_var} must have 'calendar' and 'units' atribbutes!"
        )

    nctime = dataset[time_var][:]
    time_cal = dataset[time_var].calendar
    time_unit = dataset.variables[time_var].units
    times = num2date(
        nctime, units=time_unit, calendar=time_cal, only_use_cftime_datetimes=False
    )
    times = np.array(times).astype("datetime64[ns]")
    times = times.astype("datetime64[s]").tolist()
    return times
