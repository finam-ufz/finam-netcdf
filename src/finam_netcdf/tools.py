"""NetCDF helper classes and functions"""
# pylint: disable=R0902
import fnmatch

import finam as fm
import numpy as np
from netCDF4 import num2date

ATTRS = {
    "time": {
        "axis": ("T",),
        "units": ("*since*",),  # globing for anything containing "since"
        "calendar": (
            "proleptic_gregorian",
            "gregorian",
            "julian",
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
        # "axis": ("X",),  # using this will falsely find X as lon
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
        # "axis": ("Y",),  # using this will falsely find Y as lat
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
        "standard_name": ("projection_y_coordinate",),
        "_CoordinateAxisType": ("GeoY",),
        "axis": ("Y",),
        "cartesian_axis": ("Y",),
        "grads_dim": ("y",),
    },
}


def find_axis(name, dataset):
    """
    Find axis by CF-convention hints.

    Parameters
    ----------
    name : str
        Name of the axis to find ("time", "X", "Y", "Z", "latitude", "longitude")
    dataset : netCDF4.Dataset
        The netcdf dataset to analyse.

    Returns
    -------
    set of str
        All variables that are candidates for the given axis.

    Raises
    ------
    ValueError
        If given name is not a valid axis.
    """
    if name not in ATTRS:
        raise ValueError(f"NetCDF: '{name}' not a valid axis")
    att_rules = ATTRS[name]

    def create_checker(attr):
        """
        Create a checking function to be passed to 'get_variables_by_attributes'.

        Parameters
        ----------
        attr : str
            Name of attribute that should be checked.
        """

        def checker(value):
            """Attribute value checker."""
            matches = set()
            for rule in att_rules[attr]:
                matches = matches.union(set(fnmatch.filter([str(value)], rule)))
            return any(matches)

        return checker

    # find all variables that match any rule
    axis = set()
    for att in att_rules:
        ax_vars = dataset.get_variables_by_attributes(**{att: create_checker(att)})
        axis = axis.union([v.name for v in ax_vars])
    return axis


def check_order_reversed(order):
    """
    Check if axes order is reversed

    Parameters
    ----------
    order : str
        axes order

    Returns
    -------
    bool
        True if axes order is reversed

    Raises
    ------
    ValueError
        if order is neither standard nor reversed
    """
    if order in "xyz":
        return False

    if order in "zyx":
        return True

    raise ValueError(f"NetCDF: axes order is neither standard nor reversed: '{order}'")


class DatasetInfo:
    """
    Dataset Info container.

    Parameters
    ----------
    dataset : netCDF4.Dataset
        The netcdf dataset to analyse.

    Raises
    ------
    ValueError
        If multiple time dimensions are present.
    """

    def __init__(self, dataset):
        cname = "coordinates"
        bname = "bounds"
        # may includes dims for bounds
        self.dims = set(dataset.dimensions)
        # coordinates are variables with same name as a dim
        self.coords = set(dataset.variables) & self.dims
        self.coords_with_bounds = {
            c for c in self.coords if bname in dataset[c].ncattrs()
        }
        # bound variables need to be treated separately
        self.bounds = {dataset[c].getncattr(bname) for c in self.coords_with_bounds}
        self.bounds_map = {
            c: dataset[c].getncattr(bname) for c in self.coords_with_bounds
        }
        # bnd specific dims are all dims from bounds that are not coords
        dim_sets = [set()] + [set(dataset[b].dimensions) for b in self.bounds]
        self.bounds_dims = set.union(*dim_sets) - self.coords
        # remove bound specific dims from dims
        self.dims -= self.bounds_dims
        # all relevant data in the file
        self.data = set(dataset.variables) - self.bounds - self.coords
        # all relevant data on spatial grids
        self.data_with_all_coords = {
            d for d in self.data if set(dataset[d].dimensions) <= self.coords
        }
        self.data_without_coords = {
            d for d in self.data if not (set(dataset[d].dimensions) & self.coords)
        }
        self.data_dims_map = {d: dataset[d].dimensions for d in self.data}
        # get auxiliary coordinates (given under coordinate attribute and are not dims)
        self.data_with_aux = {d for d in self.data if cname in dataset[d].ncattrs()}
        self.aux_coords_map = {
            d: dataset[d].getncattr(cname).split(" ") for d in self.data_with_aux
        }
        # needs at least one set for "union"
        aux_sets = [set()] + [set(aux) for _, aux in self.aux_coords_map.items()]
        # all auxiliary coordinates
        self.aux_coords = set.union(*aux_sets) - self.coords
        # find axis coordinates
        self.time = find_axis("time", dataset) & self.coords
        self.x = find_axis("X", dataset) & self.coords
        self.y = find_axis("Y", dataset) & self.coords
        self.z = find_axis("Z", dataset) & self.coords
        self.z_down = {}  # specify direction of z axis
        for z in self.z:
            self.z_down[z] = None  # None to indicate unknown
            if "positive" in dataset[z].ncattrs():
                self.z_down[z] = dataset[z].getncattr("positive") == "down"
        self.lon = find_axis("longitude", dataset)
        self.lat = find_axis("latitude", dataset)
        self.x -= self.lon  # treat lon separatly from x-axis
        self.y -= self.lat  # treat lat separatly from y-axis
        # state if lat/lon are valid coord axis
        self.lon_axis = bool(self.lon & self.coords)
        self.lat_axis = bool(self.lat & self.coords)
        self.all_axes = self.time | self.x | self.y | self.z
        if self.lon_axis:
            self.all_axes |= self.lon & self.coords
        if self.lat_axis:
            self.all_axes |= self.lat & self.coords
        # we need a single time dimension or none
        if len(self.time) > 1:
            raise ValueError("NetCDF: only one time axis allowed in NetCDF file.")
        self.all_static = not bool(self.time)
        if not self.all_static:
            tname = next(iter(self.time))  # get time dim name
            self.static_data = {
                d for d in self.data if tname not in dataset[d].dimensions
            }
        else:
            self.static_data = self.data
        self.temporal_data = self.data - self.static_data

    def get_axes_order(self, dims):
        """
        Determine axes order from dimension names.

        Parameters
        ----------
        dims : list of str
            Dimension names for given variable.

        Returns
        -------
        str
            axes order

        Raises
        ------
        ValueError
            If dimension is not a valid axis.
        ValueError
            If an axis is repeated.
        """
        order = ""
        for d in dims:
            if d not in self.all_axes:
                raise ValueError(
                    f"NetCDF: '{d}' is not a valid axis for a gridded data variable. "
                    "If you need this variable, slice along this axis with a fix index."
                )
            if d in self.x:
                order += "x"
            if d in self.lon & self.coords and self.lon_axis:
                order += "x"
            if d in self.y:
                order += "y"
            if d in self.lat & self.coords and self.lat_axis:
                order += "y"
            if d in self.z:
                order += "z"

        if len(set(order)) != len(order):
            raise ValueError(f"NetCDF: Data-axes are not uniquely given in '{dims}'.")

        return order


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
        data_var = data_var[time_index, ...]

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

    return info, fm.UNITS.Quantity(data_var, info.units)


def create_point_axis(cell_axis):
    """Create a point axis from a cell axis"""
    diffs = np.diff(cell_axis)
    mid = cell_axis[:-1] + diffs / 2
    first = cell_axis[0] - diffs[0] / 2
    last = cell_axis[-1] + diffs[-1] / 2
    return np.concatenate(([first], mid, [last]))


def create_time_dim(dataset, time_var):
    """returns a list of datetime.datetime objects for a given NetCDF4 time varaible"""
    if (
        "units" not in dataset[time_var].ncattrs()
        or "calendar" not in dataset[time_var].ncattrs()
    ):
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
