"""NetCDF helper classes and functions"""

# pylint: disable=R0902
import fnmatch

import finam as fm
import numpy as np
from netCDF4 import num2date
from pyproj import CRS

MASK_TBD = None
"""Indicator that the mask is to be determined."""

MESH_LOCATIONS = {"node", "edge", "face", "volume"}

MESH_CELL_TYPE_MAP = {
    "tetrahedron": fm.CellType.TETRA,
    "hexahedron": fm.CellType.HEX,
}

MESH_FACE_TYPE_MAP = {
    3: fm.CellType.TRI,
    4: fm.CellType.QUAD,
}

MESH_DIM_KIND = {1: "edge", 2: "face", 3: "volume"}

MESH_CELL_TYPE_WRITE_MAP = {
    int(fm.CellType.TETRA): (0, "tetrahedron"),
    int(fm.CellType.HEX): (2, "hexahedron"),
}

MESH_CF_ROLE = {
    "mesh_topology",
    "volume_shape_type",
    "volume_node_connectivity",
    "edge_node_connectivity",
    "face_node_connectivity",
    "volume_face_connectivity",
    "volume_edge_connectivity",
    "face_edge_connectivity",
    "volume_volume_connectivity",
    "face_face_connectivity",
    "edge_face_connectivity",
}

MESH_DIM_SPEC = {"volume_dimension", "face_dimension", "edge_dimension"}

MESH_CON_SPEC = {
    "volume_node_connectivity",
    "edge_node_connectivity",
    "face_node_connectivity",
    "volume_face_connectivity",
    "volume_edge_connectivity",
    "face_edge_connectivity",
    "volume_volume_connectivity",
    "face_face_connectivity",
    "edge_face_connectivity",
}

MESH_COORD_SPEC = {
    "node_coordinates",
    "edge_coordinates",
    "face_coordinates",
    "volume_coordinates",
}

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


def _attr_dict(dataset, var):
    attr_names = dataset[var].ncattrs()
    return {k: dataset[var].getncattr(k) for k in attr_names}


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
        self.vars = set(dataset.variables)
        # coordinates are variables with same name as a dim
        self.coords = self.vars & self.dims
        self.coords_with_bounds = {
            c for c in self.coords if bname in dataset[c].ncattrs()
        }
        # bound variables need to be treated separately
        self.bounds = {dataset[c].getncattr(bname) for c in self.coords_with_bounds}
        self.bounds_map = {
            c: dataset[c].getncattr(bname) for c in self.coords_with_bounds
        }
        self.mesh_related = {
            v
            for v in self.vars
            if (
                "cf_role" in dataset[v].ncattrs()
                and dataset[v].getncattr("cf_role") in MESH_CF_ROLE
            )
        }
        self.meshes = {
            v: _attr_dict(dataset, v)
            for v in self.mesh_related
            if dataset[v].getncattr("cf_role") == "mesh_topology"
        }
        self.mesh_data = {
            v
            for v in self.vars
            if (
                "mesh" in dataset[v].ncattrs()
                and dataset[v].getncattr("mesh") in self.meshes
            )
        }
        # bnd specific dims are all dims from bounds that are not coords
        dim_sets = [set()] + [set(dataset[b].dimensions) for b in self.bounds]
        self.bounds_dims = set.union(*dim_sets) - self.coords
        # remove bound specific dims from dims
        self.dims -= self.bounds_dims
        # all relevant data in the file
        self.data = self.vars - self.bounds - self.coords - self.mesh_related
        # all relevant data on spatial grids
        self.data_with_all_coords = {
            d
            for d in self.data
            if dataset[d].dimensions and set(dataset[d].dimensions) <= self.coords
        }
        self.data_without_coords = {
            d for d in self.data if not (set(dataset[d].dimensions) & self.coords)
        }
        self.data_dims_map = {d: dataset[d].dimensions for d in self.data}
        # get auxiliary coordinates (given under coordinate attribute and are not dims)
        self.data_with_aux = {d for d in self.data if cname in dataset[d].ncattrs()}
        self.aux_coords_map = {
            d: dataset[d].getncattr(cname).split() for d in self.data_with_aux
        }
        # needs at least one set for "union"
        aux_sets = [set()] + [set(aux) for _, aux in self.aux_coords_map.items()]
        # all auxiliary coordinates
        self.aux_coords = set.union(*aux_sets) - self.coords
        # find axis coordinates
        self.time = find_axis("time", dataset) & self.coords
        self.lon = find_axis("longitude", dataset)
        self.lat = find_axis("latitude", dataset)
        # state if lat/lon are valid coord axis
        self.lon_axis = bool(self.lon & self.coords)
        self.lat_axis = bool(self.lat & self.coords)
        self.x = find_axis("X", dataset)
        self.y = find_axis("Y", dataset)
        self.z = find_axis("Z", dataset)
        self.x -= self.lon  # treat lon separately from x-axis
        self.y -= self.lat  # treat lat separately from y-axis
        self.z_down = _set_z_down(dataset, self.z)
        axes = self.time | self.x | self.y | self.z | self.lon | self.lat
        # treat all 1d vars found by find_axis as potential axes (especially for meshes),
        # even if they don't share their name with a dimension
        self.all_axes = {a for a in axes if len(dataset[a].dimensions) == 1}
        # we need a single time dimension or none
        if len(self.time) > 1:
            raise ValueError("NetCDF: only one time axis allowed in NetCDF file.")
        self.all_static = not bool(self.time)
        if not self.all_static:
            tname = next(iter(self.time))  # get time dim name
            self.static_data = {
                d for d in self.data if tname not in dataset[d].dimensions
            }
            self.static_mesh_data = {
                d for d in self.mesh_data if tname not in dataset[d].dimensions
            }
        else:
            self.static_data = self.data
            self.static_mesh_data = self.mesh_data
        self.temporal_data = self.data - self.static_data
        self.temporal_mesh_data = self.mesh_data - self.static_mesh_data
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
                    f"NetCDF: '{d}' is not a valid coordinate axis. "
                    "If you need this variable, slice along this axis with a fix index."
                )
            if d in self.x:
                order += "x"
            if d in self.lon:
                order += "x"
            if d in self.y:
                order += "y"
            if d in self.lat:
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
    mask : :any:`Mask` value or :any:`MASK_TBD` or :any:`Ellipsis`, optional
        default masking specification of the data.

        Options:
            * :any:`Ellipsis`: use default mask from the component (default)
            * :any:`finam.Mask.FLEX`: data can have a varying mask
            * :any:`finam.Mask.NONE`: data is unmasked and given as plain numpy array
            * :any:`MASK_TBD`: constant mask will be determined from the data

    crs : str or None or Ellipsis, optional
        Coordinate reference system to force for the variable.
        Either a CRS string like `EPSG:3035`, `None` for no CRS, or :any:`Ellipsis`
        to use the CRS from the NetCDF file. Optional.

    **info_kwargs
        Optional keyword arguments to instantiate an Info object (i.e. 'grid' and 'meta')
        Used to overwrite meta data, to change units or to provide a desired grid specification.
    """

    def __init__(
        self,
        name,
        io_name=None,
        slices=None,
        static=None,
        mask=...,
        crs=...,
        **info_kwargs,
    ):
        self.name = name
        self.io_name = io_name or name
        self.slices = slices or {}
        self.static = static
        self.mask = mask
        self.crs = crs
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
        name, io_name, slices, static, mask, crs = (
            self.name,
            self.io_name,
            self.slices,
            self.static,
            self.mask,
            self.crs,
        )
        return f"Variable({name=}, {io_name=}, {slices=}, {static=}, {mask=}, {crs=}, **{self.info_kwargs})"


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
        if var.name in info.mesh_data:
            # NOTE: mesh data slicing ignored at the moment
            continue
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


def set_default_mask(variables, mask=fm.Mask.FLEX):
    """
    Set the default mask for variables if needed.

    Parameters
    ----------
    variables : list of Variable
        List of desired variables.
    mask : :any:`Mask` value or :any:`MASK_TBD`, optional
        default masking specification of the data.

        Options:
            * :any:`finam.Mask.FLEX`: data can have a varying mask (default)
            * :any:`finam.Mask.NONE`: data is unmasked and given as plain numpy array
            * :any:`MASK_TBD`: constant mask will be determined from the data
    """
    # set default mask
    for var in variables:
        if var.mask is Ellipsis:
            var.mask = mask


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


def extract_info(dataset, variable, crs, current_time=None):
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
    # NOTE: CRS info missing

    info = DatasetInfo(dataset)
    data_var = dataset[variable.name]

    # storing attributes of data_var in meta dict
    meta = {name: data_var.getncattr(name) for name in data_var.ncattrs()}
    mesh = meta.pop("mesh", None)  # remove mesh attribute if present
    location = meta.pop("location", None)  # remove location attribute if present

    # update with provided meta from variable object
    add_meta = variable.get_meta()
    if "units" in meta and "units" in add_meta:
        u1, u2 = meta["units"], add_meta["units"]
        if not fm.data.tools.equivalent_units(u1, u2):
            name = variable.name
            msg = f"NetCDF: {name} was provided with different units: {u1}, {u2}"
            raise ValueError(msg)
    meta.update(add_meta)

    if "grid" in variable.info_kwargs:
        # use provided grid from variable object if present
        grid = variable.info_kwargs["grid"]
    else:
        crs = _get_crs(dataset, variable, crs)

        if mesh is not None:
            # NOTE: warn about slices being ignored for mesh-based grids
            # NOTE: deal with transects on meshes
            grid = _create_mesh(dataset, mesh, location, info, crs)
        else:
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
                msg = (
                    f"NetCDF: {order} transect slices are not supported at the moment."
                )
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
            ax_bnds_names = [attr.get("bounds", None) for attr in axes_attrs]
            ax_bnds = [
                (None if axb is None else np.asarray(dataset.variables[axb][:]).copy())
                for axb in ax_bnds_names
            ]

            if _check_axes_uniform(axes, ax_bnds):
                dims, spacing, origin, ax_inc = _create_uniform(axes, ax_bnds)
                grid = fm.UniformGrid(
                    dims=dims,
                    spacing=spacing,
                    origin=origin,
                    data_location=fm.Location.CELLS,
                    axes_names=ax_names,
                    axes_increase=ax_inc,
                    axes_reversed=axes_reversed,
                    axes_attributes=axes_attrs,
                    crs=crs,
                )
            else:
                # NOTE: we use point-associated data here and
                #       convert it to cell-associated in the grid
                rec_axes = _create_rec_axes(axes, ax_bnds)
                grid = fm.RectilinearGrid(
                    axes=rec_axes,
                    axes_names=ax_names,
                    data_location=fm.Location.CELLS,
                    axes_reversed=axes_reversed,
                    axes_attributes=axes_attrs,
                    crs=crs,
                )

    return fm.Info(time=current_time, grid=grid, meta=meta, mask=variable.mask)


def _get_crs(dataset, variable, crs):
    """Gets the CRS from either the dataset or the variable's CRS"""

    if variable.crs is not Ellipsis:
        # Overwrite with variable's value
        crs = variable.crs

    if crs is None:
        return None

    if crs is Ellipsis:
        crs_value = None
        data_var = dataset.variables[variable.name]
        if hasattr(data_var, "grid_mapping"):
            mapping = getattr(data_var, "grid_mapping")
            crs_var = dataset.variables[mapping]
            crs_dict = {attr: getattr(crs_var, attr) for attr in crs_var.ncattrs()}
            crs_value = CRS.from_cf(crs_dict)
        return crs_value

    return CRS.from_user_input(crs)


def set_mask(info, data, dataset, variable):
    """
    Determine and set the desired mask.

    Parameters
    ----------
    info : Info
        Data info.
    data : numpy.ndarray or numpy.ma.MaskedArray
        The data slice.
    dataset : netCDF4.DataSet
        The input dataset
    variable : Variable
        The variable definition

    Returns
    -------
    data : numpy.ndarray or numpy.ma.MaskedArray
        The updated data slice.
    """
    if variable.mask is fm.Mask.FLEX:
        return data
    if variable.mask is fm.Mask.NONE:
        data_var = dataset[variable.name]
        data_var.set_always_mask(False)
        data_var.set_auto_mask(False)
        return data.filled()
    if variable.mask is None:
        # assume constant mask from first time-step
        variable.mask = data.mask
        info.mask = data.mask
        return data
    # mask is specified at this point
    msg = "NetCDF: You can not directly specify a mask for a netcdf output. Use a masking adapter."
    raise ValueError(msg)


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


def _check_axes_uniform(axes, bnds):
    """Check if all axes are uniform"""
    diffs = [
        np.diff(ax) if bd is None else bd[:, 1] - bd[:, 0] for ax, bd in zip(axes, bnds)
    ]
    return all((np.all(np.isclose(dx, dx[0])) if len(dx) > 0 else True) for dx in diffs)


def _create_uniform(axes, bnds):
    """Create inputs for uniform grid."""
    dims = [len(ax) + 1 for ax in axes]
    if None in bnds:
        diffs = [(ax[1] - ax[0] if len(ax) > 1 else 0.0) for ax in axes]
        ax_inc = [(ax[1] > ax[0] if len(ax) > 1 else True) for ax in axes]
        # if any axis has only one point, we use dx from other axes
        if all(dim == 1 for dim in dims):
            spacing = [1.0] * len(dims)  # single point -> create unit-cell
        else:
            spacing = [abs(dx) for dx in diffs]
            if dims[0] == 1:
                if dims[1] == 1:
                    spacing[0] = spacing[1] = spacing[2]
                else:
                    spacing[0] = spacing[1]
            if dims[-1] == 1:
                if dims[-2] == 1:
                    spacing[-1] = spacing[-2] = spacing[-3]
                else:
                    spacing[-1] = spacing[-2]
            if len(dims) > 1:
                if dims[1] == 1:
                    spacing[1] = spacing[0]
        origin = [np.min(ax) - sp / 2 for ax, sp in zip(axes, spacing)]
    else:
        diffs = [bd[:, 1] - bd[:, 0] for bd in bnds]
        # sometimes the bounds are not following the axis direction, so we check axis first
        ax_inc = [
            (ax[1] > ax[0] if len(ax) > 1 else dx[0] > 0) for ax, dx in zip(axes, diffs)
        ]
        spacing = [abs(dx) for dx in diffs]
        origin = [np.min(bd) for bd in bnds]
    return dims, spacing, origin, ax_inc


def _create_point_axis(cell_axis, bnd=None):
    """Create a point axis from a cell axis"""
    if bnd is None:
        diffs = np.diff(cell_axis)
        if len(diffs) == 0:  # default
            return np.array([cell_axis[0] - 0.5, cell_axis[0] + 0.5])
        mid = cell_axis[:-1] + diffs / 2
        first = cell_axis[0] - diffs[0] / 2
        last = cell_axis[-1] + diffs[-1] / 2
        return np.concatenate(([first], mid, [last]))
    # use bounds if available
    bd_inc = bnd[0, 1] > bnd[0, 0]
    ax_inc = cell_axis[1] > cell_axis[0] if len(cell_axis) > 1 else bd_inc
    # sometimes the bounds are not following the axis direction, so we check axis first
    if logical_eqv(bd_inc, ax_inc):
        return np.concatenate((bnd[:, 0], [bnd[-1, 1]]))
    # bounds should actually swap cols to follow CF-conventions
    return np.concatenate((bnd[:, 1], [bnd[-1, 0]]))


def _create_rec_axes(axes, bnds):
    return [_create_point_axis(ax, bd) for ax, bd in zip(axes, bnds)]


def _create_mesh(dataset, mesh_name, location, info, crs):
    if location not in MESH_LOCATIONS:
        msg = f"NetCDF: mesh data has invalid location: {location}."
        raise ValueError(msg)
    if mesh_name not in dataset.variables:
        msg = f"NetCDF: mesh_topology {mesh_name} not found in dataset."
        raise ValueError(msg)
    mesh_var = dataset[mesh_name]
    mesh_meta = {name: mesh_var.getncattr(name) for name in mesh_var.ncattrs()}
    if mesh_meta.get("cf_role", None) != "mesh_topology":
        msg = f"NetCDF: Variable {mesh_name} is not a mesh_topology."
        raise ValueError(msg)
    mesh_dim = mesh_meta.get("topology_dimension", None)
    if mesh_dim is None or mesh_dim not in [0, 1, 2, 3]:
        msg = f"NetCDF: mesh_topology {mesh_name} has invalid topology_dimension: {mesh_dim}."
        raise ValueError(msg)
    ax_names = mesh_meta.get("node_coordinates", "").split()
    if not ax_names or all(not ax for ax in ax_names):
        msg = f"NetCDF: mesh_topology {mesh_name} has no node_coordinates."
        raise ValueError(msg)
    order = info.get_axes_order(ax_names)
    axes_reversed = check_order_reversed(order)
    if axes_reversed:
        ax_names = ax_names[::-1]  # xyz order now
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
    dim = len(axes)
    if mesh_dim > dim:
        msg = (
            f"NetCDF: mesh_topology {mesh_name} has topology_dimension {mesh_dim} "
            f"greater than number of node_coordinates {dim}."
        )
        raise ValueError(msg)
    points = np.array(axes).T
    if location == "node":
        grid = fm.UnstructuredPoints(
            points=points,
            axes_attributes=axes_attrs,
            axes_names=ax_names,
            crs=crs,
        )
    elif location == "edge":
        if mesh_dim < 1:
            msg = (
                f"NetCDF: mesh_topology {mesh_name} has invalid "
                f"topology_dimension for edge location: {mesh_dim}."
            )
            raise ValueError(msg)
        if "edge_node_connectivity" not in mesh_meta:
            msg = (
                f"NetCDF: mesh_topology {mesh_name} missing "
                "edge_node_connectivity for edge location."
            )
            raise ValueError(msg)
        conn_name = mesh_meta["edge_node_connectivity"]
        if conn_name not in dataset.variables:
            msg = f"NetCDF: edge_node_connectivity {conn_name} not found in dataset."
            raise ValueError(msg)
        conn_var = dataset[conn_name]
        start_index = (
            conn_var.getncattr("start_index")
            if "start_index" in conn_var.ncattrs()
            else 0
        )
        connectivity = np.asarray(conn_var[:, :]).copy()
        if connectivity.shape[1] != 2:
            msg = (
                f"NetCDF: edge_node_connectivity {conn_name} has invalid number of nodes per edge: "
                f"{connectivity.shape[1]} (expected 2)."
            )
            raise ValueError(msg)
        n_cells = len(connectivity)
        cell_types = np.full(n_cells, fm.CellType.LINE, dtype=int)
        connectivity -= start_index  # convert to 0-based indexing
        # create a network
        grid = fm.UnstructuredGrid(
            points=points,
            cells=connectivity,
            cell_types=cell_types,
            data_location=fm.Location.CELLS,
            axes_attributes=axes_attrs,
            axes_names=ax_names,
            crs=crs,
        )
    elif location == "face":
        if mesh_dim < 2:
            msg = (
                f"NetCDF: mesh_topology {mesh_name} has invalid "
                f"topology_dimension for face location: {mesh_dim}."
            )
            raise ValueError(msg)
        if "face_node_connectivity" not in mesh_meta:
            msg = (
                f"NetCDF: mesh_topology {mesh_name} missing "
                "face_node_connectivity for face location."
            )
            raise ValueError(msg)
        conn_name = mesh_meta["face_node_connectivity"]
        if conn_name not in dataset.variables:
            msg = f"NetCDF: face_node_connectivity {conn_name} not found in dataset."
            raise ValueError(msg)
        conn_var = dataset[conn_name]
        start_index = (
            conn_var.getncattr("start_index")
            if "start_index" in conn_var.ncattrs()
            else 0
        )
        connectivity = conn_var[:, :].copy()
        connectivity -= start_index  # convert to 0-based indexing
        n_cells = len(connectivity)
        if connectivity.shape[1] not in [3, 4]:
            msg = (
                f"NetCDF: face_node_connectivity {conn_name} has invalid max. number of "
                f"nodes per face: {connectivity.shape[1]} (expected 3 or 4)."
            )
            raise ValueError(msg)
        if np.ma.is_masked(connectivity):
            # determine cell types from mask (len of rows: 3->triangle, 4->quad)
            con_mask = connectivity.mask
            # fill masked values with -1 for fm.Grid
            connectivity = connectivity.filled(-1)
            # determine face type from number of valid nodes (sum of non-masked values in each row)
            # 3 valid nodes -> triangle, 4 valid nodes -> quad, other -> invalid (-1)
            cell_types = [
                MESH_FACE_TYPE_MAP.get(np.sum(~con_mask[i]), -1)
                for i in range(len(connectivity))
            ]
        else:
            # umasked connectivity -> all cells have same number of nodes -> same cell type
            if connectivity.shape[1] == 3:
                cell_types = np.full(n_cells, fm.CellType.TRI, dtype=int)
            else:
                cell_types = np.full(n_cells, fm.CellType.QUAD, dtype=int)
        if any(ct == -1 for ct in cell_types):
            msg = (
                f"NetCDF: face_node_connectivity {conn_name} has invalid cell types "
                f"determined from mask (only 3 or 4 valid nodes allowed)."
            )
            raise ValueError(msg)
        # create a surface mesh
        grid = fm.UnstructuredGrid(
            points=points,
            cells=connectivity,
            cell_types=cell_types,
            data_location=fm.Location.CELLS,
            axes_attributes=axes_attrs,
            axes_names=ax_names,
            crs=crs,
        )
    else:
        # location is "volume" here
        if mesh_dim < 3:
            msg = (
                f"NetCDF: mesh_topology {mesh_name} has invalid "
                f"topology_dimension for volume location: {mesh_dim}."
            )
            raise ValueError(msg)
        if "volume_node_connectivity" not in mesh_meta:
            msg = (
                f"NetCDF: mesh_topology {mesh_name} missing "
                "volume_node_connectivity for volume location."
            )
            raise ValueError(msg)
        conn_name = mesh_meta["volume_node_connectivity"]
        if conn_name not in dataset.variables:
            msg = f"NetCDF: volume_node_connectivity {conn_name} not found in dataset."
            raise ValueError(msg)
        conn_var = dataset[conn_name]
        start_index = (
            conn_var.getncattr("start_index")
            if "start_index" in conn_var.ncattrs()
            else 0
        )
        connectivity = np.asarray(conn_var[:, :]).copy()
        connectivity -= start_index  # convert to 0-based indexing
        n_cells = len(connectivity)
        if np.ma.is_masked(connectivity):
            connectivity = connectivity.filled(
                -1
            )  # fill masked values with -1 for fm.Grid
        # cell types given by volume_shape_type attribute of mesh as integer flag values
        # defined in CF-conventions, e.g.:
        # integer Mesh3D_vol_types(nMesh3D_vol) ;
        # Mesh3D_vol_types:cf_role = "volume_shape_type" ;
        # Mesh3D_vol_types:long_name = "Specifies the shape of the individual volumes." ;
        # Mesh3D_vol_types:flag_range = 0b, 2b ;
        # Mesh3D_vol_types:flag_values = 0b, 1b, 2b ;
        # Mesh3D_vol_types:flag_meanings = "tetrahedron wedge hexahedron" ;
        if "volume_shape_type" not in mesh_meta:
            msg = (
                f"NetCDF: mesh_topology {mesh_name} missing "
                "volume_shape_type for volume location."
            )
            raise ValueError(msg)
        cell_types_var = dataset[mesh_meta["volume_shape_type"]]
        cell_types = cell_types_var[:].copy()
        # convert CF flag values/meanings to fm.CellType values (e.g. tetrahedron->TETRA, hexahedron->HEX)
        # other cell types not supported by finam -> error
        # use MESH_CELL_TYPE_MAP
        # check if var has flag_values and flag_meanings attributes
        if (
            "flag_values" not in cell_types_var.ncattrs()
            or "flag_meanings" not in cell_types_var.ncattrs()
        ):
            msg = (
                f"NetCDF: mesh_topology {mesh_name} volume_shape_type variable "
                f"{mesh_meta['volume_shape_type']} missing flag_values or flag_meanings attributes."
            )
            raise ValueError(msg)
        # create value->meaning dict for flag values
        flag_values = cell_types_var.flag_values
        flag_meanings = cell_types_var.flag_meanings.split()
        flag_cell_map = {
            fv: MESH_CELL_TYPE_MAP[fm]
            for fv, fm in zip(flag_values, flag_meanings)
            if fm in MESH_CELL_TYPE_MAP
        }
        cell_types = [flag_cell_map.get(ct, -1) for ct in cell_types]
        if any(ct == -1 for ct in cell_types):
            msg = (
                f"NetCDF: mesh_topology {mesh_name} volume_shape_type variable "
                f"{mesh_meta['volume_shape_type']} has unsupported cell types in flag_values/flag_meanings."
            )
            raise ValueError(msg)
        # create grid
        grid = fm.UnstructuredGrid(
            points=points,
            cells=connectivity,
            cell_types=cell_types,
            data_location=fm.Location.CELLS,
            axes_attributes=axes_attrs,
            axes_names=ax_names,
            crs=crs,
        )
    return grid


def create_time_dim(dataset, time_var, time_location=None):
    """returns a list of datetime.datetime objects for a given NetCDF4 time variable"""
    if (
        "units" not in dataset[time_var].ncattrs()
        or "calendar" not in dataset[time_var].ncattrs()
    ):
        msg = (
            f"NetCDF: Variable {time_var} must have 'calendar' and 'units' attributes."
        )
        raise AttributeError(msg)

    if "bounds" in dataset[time_var].ncattrs():
        # always use end of respective time-frame as output time if bounds given
        nctime = dataset[dataset[time_var].bounds][:, 2]
    elif time_location is None or np.isclose(time_location, 1):
        # assume given time stamp *is* the end of respective time-frame
        nctime = dataset[time_var][:]
    else:
        if time_location < 0 or time_location > 1:
            msg = f"NetCDF: given {time_location=} out of bounds. Should be in [0, 1]."
            raise ValueError(msg)
        rawtime = dataset[time_var][:]
        if len(rawtime) < 2:
            msg = "NetCDF: Time axis needs at least two time points to use time_location feature."
            raise ValueError(msg)
        diffs = rawtime[1:] - rawtime[:-1]
        diff = diffs[0]
        if not np.allclose(diffs, diff):
            msg = "NetCDF: Time axis needs to be uniform to use time_location feature."
            raise ValueError(msg)
        nctime = rawtime + (1 - time_location) * diff

    time_cal = dataset[time_var].calendar
    time_unit = dataset.variables[time_var].units
    times = num2date(
        nctime, units=time_unit, calendar=time_cal, only_use_cftime_datetimes=False
    )
    times = np.array(times).astype("datetime64[ns]")
    times = times.astype("datetime64[s]").tolist()
    return times


def _unique_name(base, existing):
    if base not in existing:
        return base
    idx = 1
    while f"{base}_{idx}" in existing:
        idx += 1
    return f"{base}_{idx}"


def _ensure_dimension(dataset, name, size):
    if name in dataset.dimensions:
        if len(dataset.dimensions[name]) != size:
            msg = f"NetCDF: can't add different dimension with same name: {name}"
            raise ValueError(msg)
        return
    dataset.createDimension(name, size)


def _validate_unstructured_cells(mesh_dim, cell_types):
    if mesh_dim == 0:
        if not np.all(cell_types == fm.CellType.VERTEX):
            raise ValueError("NetCDF: mesh_dim=0 requires all cell types to be VERTEX.")
        return
    if mesh_dim == 1:
        if not np.all(cell_types == fm.CellType.LINE):
            raise ValueError("NetCDF: mesh_dim=1 requires all cell types to be LINE.")
        return
    if mesh_dim == 2:
        if not np.all(np.isin(cell_types, [fm.CellType.TRI, fm.CellType.QUAD])):
            raise ValueError("NetCDF: mesh_dim=2 requires TRI or QUAD cell types.")
        return
    if mesh_dim == 3:
        if not np.all(np.isin(cell_types, [fm.CellType.TETRA, fm.CellType.HEX])):
            raise ValueError("NetCDF: mesh_dim=3 requires TETRA or HEX cell types.")
        return
    raise ValueError(f"NetCDF: invalid mesh_dim: {mesh_dim}")


def _create_unstructured_mesh(dataset, mesh_name, grid):
    mesh_dim = grid.mesh_dim
    if mesh_dim not in [0, 1, 2, 3]:
        raise ValueError(f"NetCDF: invalid mesh_dim: {mesh_dim}")

    points = np.asarray(grid.points, dtype=float)
    n_nodes, dim = points.shape
    node_dim = f"n{mesh_name}_node"
    _ensure_dimension(dataset, node_dim, n_nodes)

    # node coordinate variables
    node_coord_names = []
    axes_names = list(grid.axes_names)
    axes_attrs = list(grid.axes_attributes)
    for i in range(dim):
        ax = axes_names[i]
        base = f"{mesh_name}_node_{ax}"
        vname = _unique_name(base, dataset.variables)
        node_coord_names.append(vname)
        var = dataset.createVariable(vname, points.dtype, (node_dim,))
        attrs = dict(axes_attrs[i]) if axes_attrs[i] else {}
        attrs.setdefault("axis", "XYZ"[i])
        var.setncatts(attrs)
        var[:] = points[:, i]

    # mesh topology variable
    mesh_var = dataset.createVariable(mesh_name, "i4")
    mesh_var.cf_role = "mesh_topology"
    mesh_var.topology_dimension = mesh_dim
    mesh_var.node_coordinates = " ".join(node_coord_names)

    # no connectivity for topo-dim 0 (vertex-only meshes)
    if mesh_dim == 0:
        return {
            "mesh_name": mesh_name,
            "mesh_dim": mesh_dim,
            "node_dim": node_dim,
            "cell_dim": None,
            "kind": None,
        }

    # connectivity for edge/face/volume meshes
    cell_types = np.asarray(grid.cell_types, dtype=int)
    _validate_unstructured_cells(mesh_dim, cell_types)
    node_counts = np.asarray(grid.cell_node_counts, dtype=int)
    max_nodes = int(np.max(node_counts)) if len(node_counts) else 0

    cells = np.asarray(grid.cells, dtype=int)
    if cells.shape[1] < max_nodes:
        msg = "NetCDF: cell connectivity has fewer columns than required by cell types."
        raise ValueError(msg)

    connectivity = np.full((len(cells), max_nodes), -1, dtype=int)
    for i, cnt in enumerate(node_counts):
        if cnt > max_nodes:
            raise ValueError("NetCDF: invalid cell node count.")
        connectivity[i, :cnt] = cells[i, :cnt]

    kind = MESH_DIM_KIND[mesh_dim]
    cell_dim = f"n{mesh_name}_{kind}"
    max_nodes_dim = f"max_{mesh_name}_{kind}_nodes"
    _ensure_dimension(dataset, cell_dim, len(connectivity))
    _ensure_dimension(dataset, max_nodes_dim, max_nodes)

    fill_value = -1 if np.any(connectivity < 0) else None
    conn_name = f"{mesh_name}_{kind}_nodes"
    conn_var = dataset.createVariable(
        conn_name,
        "i4",
        (cell_dim, max_nodes_dim),
        fill_value=fill_value,
    )
    conn_var.long_name = f"Connectivity from {kind}s to nodes"
    conn_var.start_index = 0
    conn_var.cf_role = f"{kind}_node_connectivity"
    conn_var[:] = connectivity
    setattr(mesh_var, f"{kind}_node_connectivity", conn_name)

    if mesh_dim == 3:
        # volume shape types
        for ct in np.unique(cell_types):
            ct = int(ct)
            if ct not in MESH_CELL_TYPE_WRITE_MAP:
                raise ValueError("NetCDF: unsupported volume cell type for writing.")
        flag_values = []
        flag_meanings = []
        for ct in np.unique(cell_types):
            ct = int(ct)
            val, meaning = MESH_CELL_TYPE_WRITE_MAP[ct]
            flag_values.append(val)
            flag_meanings.append(meaning)

        vol_types_name = f"{mesh_name}_vol_types"
        vol_var = dataset.createVariable(vol_types_name, "i4", (cell_dim,))
        vol_var.cf_role = "volume_shape_type"
        vol_var.long_name = "Specifies the shape of the individual volumes."
        vol_var.flag_values = np.array(flag_values, dtype="i4")
        vol_var.flag_meanings = " ".join(flag_meanings)
        vol_var[:] = np.array(
            [MESH_CELL_TYPE_WRITE_MAP[int(ct)][0] for ct in cell_types], dtype="i4"
        )
        mesh_var.volume_shape_type = vol_types_name

    return {
        "mesh_name": mesh_name,
        "mesh_dim": mesh_dim,
        "node_dim": node_dim,
        "cell_dim": cell_dim,
        "kind": kind,
    }


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
    variables : list of Variable
        Variable informations.
    global_attrs : dict
        global attributes for the NetCDF file inputted by the user

    Raises
    ------
    ValueError
        If there is a duplicated output parameter variable.
    ValueError
        If the names of the XYZ coordinates do not match for all variables.
    ValueError
        If a input coordinate is not in grid_info.axes_name variables.
    ValueError
        If unstructured grids are malformed or unsupported.
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
    else:
        non_static = [var.name for var in variables if not var.static]
        if any(non_static):
            msg = f"NetCDF: dataset has no time but some variables are not static: {non_static}"
            raise ValueError(msg)

    mesh_registry = []

    def _get_mesh_info(grid):
        for entry in mesh_registry:
            if grid.compatible_with(entry["grid"], check_location=False):
                return entry
        mesh_name = _unique_name(f"mesh{grid.mesh_dim}d", dataset.variables)
        mesh_info = _create_unstructured_mesh(dataset, mesh_name, grid)
        entry = {"grid": grid, **mesh_info}
        mesh_registry.append(entry)
        return entry

    crs_registry = []

    for var in variables:
        grid = in_infos[var.io_name].grid
        crs_index = -1
        if grid.crs is not None:
            crs = CRS.from_user_input(grid.crs)
            try:
                crs_index = crs_registry.index(crs)
            except ValueError:
                crs_index = len(crs_registry)
                crs_registry.append(crs)

        if isinstance(grid, fm.data.StructuredGrid):
            axes_names = (
                tuple(reversed(grid.axes_names))
                if grid.axes_reversed
                else tuple(grid.axes_names)
            )

            for i, ax in enumerate(axes_names):
                if ax in dataset.variables:
                    # check if existing axes is same as this one
                    ax1, ax2 = dataset[ax][:], grid.data_axes[i]
                    if np.size(ax1) == np.size(ax2) and np.allclose(ax1, ax2):
                        continue
                    raise ValueError("NetCDF: can't add different axes with same name.")
                dataset.createDimension(ax, len(grid.data_axes[i]))
                dataset.createVariable(ax, grid.data_axes[i].dtype, (ax,))
                dataset[ax].setncatts(grid.axes_attributes[i])
                dataset[ax].setncattr("axis", "XYZ"[i])
                dataset[ax][:] = grid.data_axes[i]
                # add axis bounds if data location is cells

            dim = (time_var,) * (not var.static) + axes_names
            dtype = np.asanyarray(in_data[var.io_name].magnitude).dtype
            ncvar = dataset.createVariable(var.name, dtype, dim)
            meta = in_infos[var.io_name].meta
            ncvar.setncatts({n: str(v) if n == "units" else v for n, v in meta.items()})
            if grid.crs is not None:
                ncvar.setncatts({"grid_mapping": f"crs_{crs_index}"})

        elif isinstance(grid, fm.UnstructuredGrid):
            mesh_info = _get_mesh_info(grid)
            location = (
                "node"
                if grid.data_location == fm.Location.POINTS
                else mesh_info["kind"]
            )
            if location is None:
                location = "node"
            dim_name = (
                mesh_info["node_dim"] if location == "node" else mesh_info["cell_dim"]
            )
            dim = (time_var,) * (not var.static) + (dim_name,)
            dtype = np.asanyarray(in_data[var.io_name].magnitude).dtype
            ncvar = dataset.createVariable(var.name, dtype, dim)
            meta = dict(in_infos[var.io_name].meta)
            meta["mesh"] = mesh_info["mesh_name"]
            meta["location"] = location
            ncvar.setncatts({n: str(v) if n == "units" else v for n, v in meta.items()})
            if grid.crs is not None:
                ncvar.setncatts({"grid_mapping": f"crs_{crs_index}"})

        else:
            msg = f"NetCDF: {var.name} is not given on a supported grid."
            raise ValueError(msg)

    for i, crs in enumerate(crs_registry):
        crs_var = dataset.createVariable(f"crs_{i}", "i4")

        # Write WKT
        crs_var.spatial_ref = crs.to_wkt()
        crs_var.crs_wkt = crs.to_wkt()

        # Write CF projection parameters
        cf = crs.to_cf()
        for key, value in cf.items():
            setattr(crs_var, key, value)

        # EPSG code
        if crs.to_epsg() is not None:
            crs_var.epsg_code = f"EPSG:{crs.to_epsg()}"
