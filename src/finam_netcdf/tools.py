"""NetCDF helper classes and functions"""
# pylint: disable=R0902
import fnmatch

import finam as fm
import numpy as np
from netCDF4 import num2date

Z_STD_NAME_POSITIVE = {
    "altitude": "up",
    "atmosphere_ln_pressure_coordinate": "down",
    "atmosphere_sigma_coordinate": "down",
    "atmosphere_hybrid_sigma_pressure_coordinate": "down",
    "atmosphere_sigma": "down",
    "ocean_sigma_coordinate": "up",
    "ocean_s_coordinate": "down",
    "ocean_s_coordinate_g1": "down",
    "ocean_s_coordinate_g2": "down",
    "ocean_s_coordinate_g1_threshold": "down",
    "ocean_s_coordinate_g2_threshold": "down",
    "ocean_sea_water_sigma": "down",
    "ocean_sea_water_sigma_theta": "down",
    "ocean_sea_water_potential_temperature": "down",
    "ocean_sea_water_salinity": "down",
    "ocean_density": "down",
    "ocean_sigma": "down",
    "ocean_isopycnal_coordinate": "down",
    "ocean_isopycnal_potential_density": "down",
    "ocean_isopycnal_theta": "down",
    "ocean_isopycnal_sigma": "down",
    "ocean_layer": "down",
    "ocean_sigma_z": "down",
    "ocean_sigma_theta": "down",
    "ocean_double_sigma_coordinate": "down",
    "ocean_double_sigma_coordinate_g1": "down",
    "ocean_double_sigma_coordinate_g2": "down",
    "ocean_double_sigma_coordinate_g1_threshold": "down",
    "ocean_double_sigma_coordinate_g2_threshold": "down",
    "ocean_z_coordinate": "down",
    "ocean_z_coordinate_g1": "down",
    "ocean_z_coordinate_g2": "down",
    "ocean_z_coordinate_g1_threshold": "down",
    "ocean_z_coordinate_g2_threshold": "down",
    "height": "up",
    "height_above_geopotential_surface": "up",
    "height_above_reference_ellipsoid": "up",
    "height_above_sea_floor": "up",
    "depth": "down",
    "depth_below_geoid": "down",
    "depth_below_sea_floor": "down",
}

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
        "standard_name": tuple(Z_STD_NAME_POSITIVE),
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


def logical_eqv(a, b):
    """Logical equivalence."""
    return (a and b) or (not a and not b)


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
    Check if axes order is reversed.

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
    if order in "xyz" or order == "xz":
        return False

    if order in "zyx" or order == "zx":
        return True

    raise ValueError(f"NetCDF: axes order is neither standard nor reversed: '{order}'")


def is_transect(order):
    """
    Check if axes order is defining a transect.

    Parameters
    ----------
    order : str
        axes order

    Returns
    -------
    bool
        True if axes order is "yz", "xz", "zy" or "zx"
    """
    return order in ["yz", "xz", "zy", "zx"]


def _set_z_down(dataset, zvars):
    z_down = {}  # specify direction of z axis
    for z in zvars:
        z_down[z] = None  # None to indicate unknown
        if "positive" in dataset[z].ncattrs():
            z_down[z] = dataset[z].getncattr("positive") == "down"
        elif "standard_name" in dataset[z].ncattrs():
            std_name = dataset[z].getncattr("standard_name")
            if std_name in Z_STD_NAME_POSITIVE:
                z_down[z] = Z_STD_NAME_POSITIVE[std_name] == "down"
    return z_down


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
        self.z_down = _set_z_down(dataset, self.z)
        self.lon = find_axis("longitude", dataset)
        self.lat = find_axis("latitude", dataset)
        self.x -= self.lon  # treat lon separately from x-axis
        self.y -= self.lat  # treat lat separately from y-axis
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
        self.data_spatial_dims_map = {
            d: [i for i in v if i not in self.time]
            for d, v in self.data_dims_map.items()
        }

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


class Variable:
    """
    Specifications for a NetCDF variable.

    Parameters
    ----------

    name : str
        Variable name in the NetCDF file.
    io_name : str, optional
        Desired name of the respective Input/Output in the FINAM component.
        Will be the variable name by default.
    slices : dict of str, int, optional
        Dictionary for fixed coordinate indices (e.g. {'time': 0})
    static : bool or None, optional
        Flag indicating static data. If None, this will be determined.
        Writer will interprete None as False.
        Default: None
    **info_kwargs
        Optional keyword arguments to instantiate an Info object (i.e. 'grid' and 'meta')
        Used to overwrite meta data, to change units or to provide a desired grid specification.
    """

    def __init__(self, name, io_name=None, slices=None, static=None, **info_kwargs):
        self.name = name
        self.io_name = io_name or name
        self.slices = slices or {}
        self.static = static
        self.info_kwargs = info_kwargs

    def get_meta(self):
        """Get the meta-data dictionary of this variable."""
        meta = self.info_kwargs.get("meta", {})
        meta.update(
            {
                k: v
                for k, v in self.info_kwargs.items()
                if k not in ["time", "grid", "meta"]
            }
        )
        return meta

    def __repr__(self):
        name, io_name, slices, static = (
            self.name,
            self.io_name,
            self.slices,
            self.static,
        )
        return (
            f"Variable({name=}, {io_name=}, {slices=}, {static=}, **{self.info_kwargs})"
        )


def create_variable_list(variables):
    """
    Create a list of Variable instances.

    Parameters
    ----------
    variables : list of str or Variable
        List containing Variable instances or names.

    Returns
    -------
    list of Variable
        List containing only Variable instances.
    """
    return [var if isinstance(var, Variable) else Variable(var) for var in variables]


def extract_variables(dataset, variables=None, only_static=False):
    """
    Extract the variable information from a dataset following CF convention.

    Parameters
    ----------
    dataset : netCDF4.Dataset
        Opened NetCDF dataset.
    variables : list of Variable or str, optional
        List of desired variables given by name or a :class:`Variable` instance.
        By default, all variables present in the NetCDF file.
    only_static : bool, optional
        Only provide static variables, or variables with a fixed time slice.
        Default: False

    Returns
    -------
    variables : list of Variable
        Variables information.
    """

    info = DatasetInfo(dataset)
    if variables is None:
        variables = create_variable_list(info.static_data if only_static else info.data)
    else:
        variables = create_variable_list(variables)
    # check if all variables are present
    if not set(v.name for v in variables) <= info.data:
        miss = set(v.name for v in variables) - info.data
        msg = f"NetCDF: some variables are not present in the file: {miss}"
        raise ValueError(msg)
    # check for static data
    tname = None if info.all_static else next(iter(info.time))
    for var in variables:
        if info.all_static:
            if var.static is not None and not var.static:
                msg = f"NetCDF: Variable wasn't flagged static but is: {var.name}"
                raise ValueError(msg)
            var.static = True
        else:
            static = var.name in info.static_data or tname in var.slices
            if var.static is not None and not logical_eqv(var.static, static):
                msg = f"NetCDF: Variable has a wrong static flag: {var.name}"
            var.static = static
    if only_static and not info.all_static:
        if not all(var.static for var in variables):
            temp = [var.name for var in variables if not var.static]
            msg = f"NetCDF: Some variables are not static but should: {temp}"
            raise ValueError(msg)
    # check if all variables have correct dims and slices
    for var in variables:
        slice_dims = set(var.slices)
        all_dims = set(info.data_dims_map[var.name])
        if not slice_dims <= all_dims:
            miss = slice_dims - all_dims
            msg = f"NetCDF: Variable {var.name} doesn't have required dimensions for slicing: {miss}"
            raise ValueError(msg)
        if (
            var.name not in info.data_with_all_coords
            and not all_dims - slice_dims <= info.coords
        ):
            miss = all_dims - slice_dims - info.coords
            msg = f"NetCDF: Variable {var.name} misses coordinates: {miss}."
            raise ValueError(msg)
    return variables


def extract_time(dataset):
    """
    Extract the time coordinate name from a dataset following CF convention.

    Parameters
    ----------
    dataset : netCDF4.Dataset
        Opened NetCDF dataset.

    Returns
    -------
    time : str or None
        Name of time coordinate if present.
    """
    info = DatasetInfo(dataset)
    return None if info.all_static else next(iter(info.time))


def extract_info(dataset, variable, current_time=None):
    """Extracts the Info object for the selected variable.

    Parameters
    ----------
    dataset : netCDF4.DataSet
        The input dataset
    variable : Variable
        The variable definition
    current_time : datetime.datetime or None
        Current time for the Info object.
    """

    info = DatasetInfo(dataset)
    data_var = dataset[variable.name]

    # storing attributes of data_var in meta dict
    meta = {name: data_var.getncattr(name) for name in data_var.ncattrs()}

    # checks if axes were reversed or not
    ax_names = [
        ax
        for ax in info.data_spatial_dims_map[variable.name]
        if ax not in variable.slices
    ]
    order = info.get_axes_order(ax_names)
    axes_reversed = check_order_reversed(order)
    if axes_reversed:
        ax_names = ax_names[::-1]  # xyz order now

    # this needs some work with the respective grid to be created correctly
    if is_transect(order):
        msg = f"NetCDF: {order} transect slices are not supported at the moment."
        raise ValueError(msg)

    # getting coordinates data
    axes = [np.asarray(dataset.variables[ax][:]).copy() for ax in ax_names]
    # _FillValue and missing_value not allowed for coordinates
    axes_attrs = [
        {
            name: dataset.variables[ax].getncattr(name)
            for name in dataset.variables[ax].ncattrs()
            if name not in ["_FillValue", "missing_value"]
        }
        for ax in ax_names
    ]
    if "grid" in variable.info_kwargs:
        # use provided grid from variable object if present
        grid = variable.info_kwargs["grid"]
    else:
        # note: we use point-associated data here.
        grid = fm.RectilinearGrid(
            axes=[_create_point_axis(ax) for ax in axes],
            axes_names=ax_names,
            data_location=fm.Location.CELLS,
            axes_reversed=axes_reversed,
            axes_attributes=axes_attrs,
        )

    # getting current time
    time = None if variable.static else current_time

    # update with provided meta from variable object
    meta.update(variable.get_meta())

    return fm.Info(time=time, grid=grid, meta=meta)


def extract_data(dataset, variable, time_var=None, time_index=None):
    """Extracts the Info object for the selected variable.

    Parameters
    ----------
    dataset : netCDF4.DataSet
        The input dataset
    variable : Variable
        The variable definition
    time_var : str or None
        Name of time coordinate if present.
    time_index : int or None
        Selected time index if data is not static.

    Returns
    -------
    data : numpy.ndarray or numpy.ma.MaskedArray
        The data slice.
    """
    data_var = dataset[variable.name]
    slices = variable.slices
    if not variable.static:
        slices[time_var] = time_index
    return data_var[_get_slice(data_var.dimensions, slices)]


def _get_slice(dims, slices):
    return tuple(slices.get(d, slice(None)) for d in dims)


def _create_point_axis(cell_axis):
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
            f"Variable {time_var} must have 'calendar' and 'units' attributes."
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


def create_nc_framework(
    dataset,
    time_var,
    start_date,
    time_freq,
    in_infos,
    in_data,
    variables,
    global_attrs,
):
    """
    Creates a NetCDF file for given data.

    Parameters
    ----------
    dataset : netCDF4._netCDF4.Dataset
        empty NetCDF file
    time_var : str or None
        name of the time variable
    start_date : datetime.datetime
        starting time
    time_freq : datetime.datetime | str
        time stepping
    in_infos : dict
        grid data and units for each output variable
    in_data : dict
        array data and units for each output variable
    variables : list
        Layer information for each variable:
        Layer(var=--, xyz=(--, --, --), fixed={--}, static=--))
    global_attrs : dict
        global attributes for the NetCDF file inputed by the user

    Raises
    ------
    ValueError
        If there is a duplicated output parameter varaible.
    ValueError
        If the names of the XYZ coordinates do not match for all variables.
    ValueError
        If a input coordinates not in grid_info.axes_name variables.
    """
    # adding general user input attributes if any
    dataset.setncatts(global_attrs)

    if time_var is not None:
        # creating time dim and var
        dataset.createDimension(time_var, None)
        t_var = dataset.createVariable(time_var, np.float64, (time_var,))

        if isinstance(time_freq, str):
            freq = time_freq
        elif time_freq.days != 0:
            freq = "days"
        elif time_freq.seconds // 3600 != 0:
            freq = "hours"
        elif (time_freq.seconds // 60) % 60 != 0:
            freq = "minutes"
        else:
            freq = "seconds"

        t_var.units = f"{freq} since {start_date}"
        t_var.calendar = "standard"

    for var in variables:
        grid_info = in_infos[var.io_name].grid
        for i, ax in enumerate(grid_info.axes_names):
            if ax in dataset.variables:
                # check if existing axes is same as this one
                ax1, ax2 = dataset[ax][:], grid_info.data_axes[i]
                if np.size(ax1) == np.size(ax2) and np.allclose(ax1, ax2):
                    continue
                raise ValueError("NetCDF: can't add different axes with same name.")
            dataset.createDimension(ax, len(grid_info.data_axes[i]))
            dataset.createVariable(ax, grid_info.data_axes[i].dtype, (ax,))
            dataset[ax].setncatts(grid_info.axes_attributes[i])
            dataset[ax].setncattr("axis", "XYZ"[i])
            dataset[ax][:] = grid_info.data_axes[i]
            # add axis bounds if data location is cells

        dim = (time_var,) * (not var.static) + tuple(grid_info.axes_names)
        dtype = np.asanyarray(in_data[var.io_name].magnitude).dtype
        ncvar = dataset.createVariable(var.name, dtype, dim)
        meta = in_infos[var.io_name].meta
        ncvar.setncatts({n: str(v) if n == "units" else v for n, v in meta.items()})
