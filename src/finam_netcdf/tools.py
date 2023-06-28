"""NetCDF helper classes and functions"""
import warnings

import finam as fm
import numpy as np
from netCDF4 import num2date

ATTRS = {
    "time": {
        "axis": ("T",),
        "units": ("since",),
        "calendar": (
            "proleptic_gregorian",
            "gregorian" "julian",
            "standard",
            "noleap",
            "365_day",
            "all_leap",
            "366_day",
            "360_day",
            "none",
        ),
        "standard_name": ("time",),
        "long_name": ("time",),
        "_CoordinateAxisType": ("Time",),
        "cartesian_axis": ("T",),
        "grads_dim": ("t",),
    },
    "longitude": {
        "axis": "X",
        "units": (
            "degrees_east",
            "degree_east",
            "degree_E",
            "degrees_E",
            "degreeE",
            "degreesE",
        ),
        "standard_name": ("longitude",),
        "long_name": ("longitude",),
        "_CoordinateAxisType": ("Lon",),
    },
    "latitude": {
        "axis": ("Y",),
        "units": (
            "degrees_north",
            "degree_north",
            "degree_N",
            "degrees_N",
            "degreeN",
            "degreesN",
        ),
        "standard_name": ("latitude",),
        "long_name": ("latitude",),
        "_CoordinateAxisType": ("Lat",),
    },
    "Z": {
        "axis": ("Z",),
        "standard_name": (
            "level",
            "pressure level",
            "depth",
            "height",
            "vertical level",
            "elevation",
            "altitude",
        ),
        "long_name": (
            "level",
            "pressure level",
            "depth",
            "height",
            "vertical level",
            "elevation",
            "altitude",
        ),
        "positive": ("up", "down"),
        "_CoordinateAxisType": (
            "GeoZ",
            "Height",
            "Pressure",
        ),
        "cartesian_axis": ("Z",),
        "grads_dim": ("z",),
    },
    "X": {
        "standard_name": ("projection_x_coordinate",),
        "_CoordinateAxisType": ("GeoX",),
        "axis": ("X",),
        "cartesian_axis": ("X",),
        "grads_dim": ("x",),
    },
    "Y": {
        "standard_name": "projection_y_coordinate",
        "_CoordinateAxisType": ("GeoY",),
        "axis": ("Y",),
        "cartesian_axis": ("Y",),
        "grads_dim": ("y",),
    },
}


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
    Extracts the layer information from a dataset following CF convention
    """
    layers = []
    var_list = []
    time_var = None
    x_var = None
    y_var = None
    z_var = None
    fixed = None

    for var, data in dataset.variables.items():
        if len(data.dimensions) > 4:
            raise ValueError(
                f"Variable {data.name} has more than 4 possible dimensions (T, Z, Y, X)."
            )
        elif var in dataset.dimensions.keys():
            time_var, check_var = _check_var_attr(time_var, data, var, ATTRS["time"])
            time_var = _check_var(time_var, data.name, check_var)

            z_var, check_var = _check_var_attr(z_var, data, var, ATTRS["Z"])
            z_var = _check_var(z_var, data.name, check_var)

            x_var, check_var = _check_var_attr(x_var, data, var, ATTRS["longitude"])
            x_var = _check_var(x_var, data.name, check_var)
            x_var, check_var = _check_var_attr(x_var, data, var, ATTRS["X"])
            x_var = _check_var(x_var, data.name, check_var)

            y_var, check_var = _check_var_attr(y_var, data, var, ATTRS["latitude"])
            y_var = _check_var(y_var, data.name, check_var)
            y_var, check_var = _check_var_attr(y_var, data, var, ATTRS["Y"])
            y_var = _check_var(y_var, data.name, check_var)
        else:
            var_list.append(data.name)

    if time_var is None:
        warnings.warn(f"{time_var}=None. Time was assigned as a static variable.")

    xyz = tuple(v for v in [x_var, y_var, z_var] if v is not None)
    if len(xyz) < 2:
        raise ValueError(
            f"CF conventions are not met or coordinates are missing. Input (X,Y) NetCDF coordinates: {xyz}."
        )
    elif len(data.dimensions) == 4 and z_var is None:
        raise ValueError(
            "Input NetCDF coordinate Z does not comply with CF conventions."
        )
    elif len(xyz) == 3:
        fixed = {}
        fixed[z_var] = 0
        warning_message = (
            f"Z dimension was assigned as a fixed variable: fixed = {fixed}."
        )
        warnings.warn(warning_message)

    for var in var_list:
        check_static = False
        if len(dataset[var].shape) < 3:
            check_static = True

        if len(xyz) == 3:
            layers.append(
                Layer(
                    var,
                    xyz[0:2],
                    fixed=fixed,
                    static=time_var is None or check_static is True,
                )
            )
        else:
            layers.append(
                Layer(var, xyz, static=time_var is None or check_static is True)
            )

    return time_var, z_var, layers


def _check_var(old, new, check_var=True):
    if check_var == True:
        if old is None or old == new:
            return new
        raise ValueError(f"Axis already defined as {old}. Found second axis {new}.")
    else:
        return old


def _check_var_attr(variable, data, name, attributes):
    if variable is not None:
        return variable, False
    else:
        for attribute_name, attribute_values in attributes.items():
            if attribute_name in data.ncattrs():
                attribute_value = getattr(data, attribute_name)
                if attribute_value in attribute_values:
                    return name, True
        return None, False


def extract_grid(dataset, layer, time_var, z_var, time_index=None, current_time=None):
    """Extracts a 2D data array from a dataset

    Parameters
    ----------
    dataset : netCDF4.DataSet
        The input dataset
    layer : Layer
        The layer definition
    time_var: str
        Time variable string
    z_var: str
        Z variable string
    time_index : int, optional
        index of current time
    current_time: datetime.datetime
        (YYYY, M, D, H, S)
    """

    data_var = dataset[layer.var]

    for ax in layer.xyz:
        if ax not in data_var.dimensions:
            raise ValueError(
                f"Dimension {ax} not available for NetCDF variable {layer.var}"
            )

    # storing attributes in meta dict
    meta = {}
    for name in data_var.ncattrs():
        meta[name] = getattr(data_var, name)

    # gets data for each time step as np.array
    if time_index == None:
        new_tuple = dataset[layer.var].dimensions
    else:
        data_var = data_var[time_index, ...]
        data_var = np.array(data_var.filled(np.nan))

        # creates a new_tuple without time and Z dimension
        index_time = dataset[layer.var].dimensions.index(time_var)
        new_tuple = (
            dataset[layer.var].dimensions[:index_time]
            + dataset[layer.var].dimensions[index_time + 1 :]
        )
        if z_var is not None:
            index_zc = new_tuple.index(z_var)
            new_tuple = new_tuple[:index_zc] + new_tuple[index_zc + 1 :]

    # checks if order is xyz, if not transpose
    axes_reversed = new_tuple != layer.xyz
    if new_tuple != layer.xyz and new_tuple != layer.xyz[::-1]:
        raise ValueError(
            f"axes={new_tuple} order not valid. Must be {layer.xyz} or {layer.xyz[::-1]}"
        )

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
