"""NetCDF helper classes and functions"""
import finam as fm
import numpy as np
from netcdf_info import DatasetInfo, check_order_reversed
from netCDF4 import num2date


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

    variables = DatasetInfo(dataset).data_dims_map
    time_var = DatasetInfo(dataset).time

    if time_var == set():
        time_var = None
    else:
        time_var = time_var.pop()

    # extracting names of coordinates XYZ in the original NetCDF
    for var, dims in variables.items():
        if time_var in dims:
            xyz = tuple(value for value in dims if value not in time_var)
            layers.append(Layer(var, xyz))
        else:
            xyz = dims
            layers.append(Layer(var, xyz, static=True))

    return time_var, layers


def extract_grid(dataset, layer, time_var, time_index=None, current_time=None):
    """Extracts a 2D data array from a dataset

    Parameters
    ----------
    dataset : netCDF4.DataSet
        The input dataset
    layer : Layer
        The layer definition
    time_var: str
        Time variable string
    time_index : int, optional
        index of current time
    current_time: datetime.datetime
        (YYYY, M, D, H, S)
    """

    data_var = dataset[layer.var]

    # storing attributes of data_var in meta dict
    meta = {}
    for name in data_var.ncattrs():
        meta[name] = getattr(data_var, name)

    # gets the data for each time step as np.array if time is not None
    if time_index != None:
        data_var = data_var[time_index, ...]
        data_var = np.array(data_var.filled(np.nan))

    # checks if order is xyz, if not transpose
    data_info = DatasetInfo(dataset)
    order = data_info.get_axes_order(layer.xyz)
    axes_reversed = check_order_reversed(order)

    # getting coordinates data
    axes = [np.asarray(dataset.variables[ax][:]).copy() for ax in layer.xyz]
    axes_attrs = [dataset.variables[ax].ncattrs() for ax in layer.xyz]

    # check if axes increasing and flip inplace
    axes_increase = fm.data.check_axes_monotonicity(axes)

    # calculate properties of uniform grids
    spacing = fm.data.check_axes_uniformity(axes)
    is_uniform = not any(np.isnan(spacing))

    # note: we use point-associated data here.
    if is_uniform:
        origin = [ax[0] for ax in axes]
        dims = [len(ax) + 1 for ax in axes]
        grid = fm.UniformGrid(
            dims,
            axes_names=layer.xyz,
            spacing=tuple(spacing),
            origin=tuple(o - 0.5 * s for o, s in zip(origin, spacing)),
            data_location=fm.Location.CELLS,
            axes_reversed=axes_reversed,
            axes_increase=axes_increase,
            axes_attributes=axes_attrs,
        )
    else:
        grid = fm.RectilinearGrid(
            axes=[create_point_axis(ax) for ax in axes],
            axes_names=layer.xyz,
            data_location=fm.Location.CELLS,
            axes_reversed=axes_reversed,
            axes_increase=axes_increase,
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
