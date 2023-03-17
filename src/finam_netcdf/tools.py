"""NetCDF helper classes and functions"""
import copy

import finam as fm
import numpy as np
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
    Extracts layer information from a dataset
    """
    # needed variables
    layers = []
    var_list = []
    time_var = None
    x_var = None
    y_var = None
    z_var = None

    for var, data in dataset.variables.items():
        # getting measured variables with at least time,lon,lat info
        if len(data.dimensions) > 2:
            var_list.append(data.name)
        # assigning axis
        else:
            if "axis" in data.ncattrs():
                ax = data.axis
                if ax == "T":
                    time_var = _check_var(time_var, data.name)
                elif ax == "X":
                    x_var = _check_var(x_var, data.name)
                elif ax == "Y":
                    y_var = _check_var(y_var, data.name)
                elif ax == "Z":
                    z_var = _check_var(z_var, data.name)
            else:
                if "calendar" in data.ncattrs():
                    time_var = _check_var(time_var, data.name)

    # creating coords tuple
    xyz = tuple(v for v in [x_var, y_var, z_var] if v is not None)
    # appending to layers
    for var in var_list:
        layers.append(Layer(var, xyz, static=time_var is None))

    return time_var, layers


def _check_var(old, new):
    if old is None or old == new:
        return new
    raise ValueError(f"Axis already defined as {old}. Found second axis {new}.")


def extract_grid(dataset, layer, fixed=None):
    """Extracts a 2D data array from a dataset

    Parameters
    ----------
    dataset : netCDF4.DataSet
        The input dataset
    layer : Layer
        The layer definition
    fixed : dict of str, int, optional
        Fixed variables with their indices
    """
    var_data = dataset[layer.var][:]  # xarray values for temp
    xyz_data = [dataset.variables[ax] for ax in layer.xyz]

    # DON'T GET THIS ISEL PART! ----------------
    # ncdata = variable.isel(layer.fixed if fixed is None else dict(layer.fixed, **fixed))

    if len(var_data.dimensions) > 3:
        raise ValueError(f"NetCDF variable {layer.var} has more than 3 dimensions")
    if len(var_data.dimensions) != len(layer.xyz):
        raise ValueError(
            f"NetCDF variable {layer.var} has a different number of dimensions than given axes"
        )

    for ax in layer.xyz:
        if ax not in var_data.dimensions:
            raise ValueError(
                f"Dimension {ax} not available for NetCDF variable {layer.var}"
            )

    # copying axes values into a np array
    axes = [np.array(ax[:].copy()) for ax in xyz_data]

    # re-order axes to xyz_data
    axes = axes.transpose(*layer.xyz)

    # flip to make all axes increase
    # axes are automatically flipped when check_axes_monotonicity returns FALSE
    fm.data.check_axes_monotonicity(axes)

    # calculate properties of uniform grids
    spacing = fm.data.check_axes_uniformity(axes)
    origin = [ax[0] for ax in axes]
    is_uniform = not any(np.isnan(spacing))

    # note: we use point-associated data here.
    if is_uniform:
        dims = [len(ax) + 1 for ax in axes]
        grid = fm.UniformGrid(
            dims,
            axes_names=layer.xyz,
            spacing=tuple(spacing),
            origin=tuple(o - 0.5 * s for o, s in zip(origin, spacing)),
            data_location=fm.Location.CELLS,
        )
    else:
        point_axes = [create_point_axis(ax) for ax in axes]
        grid = fm.RectilinearGrid(
            point_axes, axes_names=layer.xyz, data_location=fm.Location.CELLS
        )

    meta = {}
    for name in dataset.ncattrs():
        meta[name] = getattr(dataset, name)
    meta = copy.copy(meta)

    # re-insert the time dimension
    times = None
    if not layer.static and "time" in dataset.dimensions:
        # variables needed
        nctime = dataset["time"][:]
        time_cal = dataset["time"].calendar
        time_unit = dataset.variables["time"].units

        # convert time fom cftime.real_datetime to datetime64 format
        times = num2date(
            nctime, units=time_unit, calendar=time_cal, only_use_cftime_datetimes=False
        )
        times = np.array(times)
        times = times.astype("datetime64[ns]")
        times = fm.data.to_datetime(times)

    info = fm.Info(time=times, grid=grid, meta=meta)

    #Are final changes in fm.UNITS to handle Netcdf4 needed? not sure what to do - What about var_data?
    return info, fm.UNITS.Quantity(axes, info.units)

def create_point_axis(cell_axis):
    """Create a point axis from a cell axis"""
    diffs = np.diff(cell_axis)
    mid = cell_axis[:-1] + diffs / 2
    first = cell_axis[0] - diffs[0] / 2
    last = cell_axis[-1] + diffs[-1] / 2
    return np.concatenate(([first], mid, [last]))
