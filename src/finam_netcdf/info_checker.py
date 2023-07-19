import fnmatch

from netCDF4 import Dataset

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


def find_axis(name, ds):
    """
    Find axis by CF-convention hints.

    Parameters
    ----------
    name : str
        Name of the axis to find ("time", "X", "Y", "Z", "latitude", "longitude")
    ds : netCDF4.Dataset
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
        ax_vars = ds.get_variables_by_attributes(**{att: create_checker(att)})
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
    ds : netCDF4.Dataset
        The netcdf dataset to analyse.

    Raises
    ------
    ValueError
        If multiple time dimensions are present.
    """

    def __init__(self, ds):
        cname = "coordinates"
        bname = "bounds"
        # may includes dims for bounds
        self.dims = set(ds.dimensions)
        # coordinates are variables with same name as a dim
        self.coords = set(ds.variables) & self.dims
        self.coords_with_bounds = {c for c in self.coords if bname in ds[c].ncattrs()}
        # bound variables need to be treated separately
        self.bounds = {ds[c].getncattr(bname) for c in self.coords_with_bounds}
        self.bounds_map = {c: ds[c].getncattr(bname) for c in self.coords_with_bounds}
        # bnd specific dims are all dims from bounds that are not coords
        dim_sets = [set()] + [set(ds[b].dimensions) for b in self.bounds]
        self.bounds_dims = set.union(*dim_sets) - self.coords
        # remove bound specific dims from dims
        self.dims -= self.bounds_dims
        # all relevant data in the file
        self.data = set(ds.variables) - self.bounds - self.coords
        # all relevant data on spatial grids
        self.data_with_all_coords = {
            d for d in self.data if set(ds[d].dimensions) <= self.coords
        }
        self.data_without_coords = {
            d for d in self.data if not (set(ds[d].dimensions) & self.coords)
        }
        self.data_dims_map = {d: ds[d].dimensions for d in self.data}
        # get auxiliary coordinates (given under coordinate attribute and are not dims)
        self.data_with_aux = {d for d in self.data if cname in ds[d].ncattrs()}
        type(self).aux_coords_map = {
            d: ds[d].getncattr(cname).split(" ") for d in self.data_with_aux
        }
        # needs at least one set for "union"
        aux_sets = [set()] + [set(aux) for _, aux in self.aux_coords_map.items()]
        # all auxiliary coordinates
        self.aux_coords = set.union(*aux_sets) - self.coords
        # find axis coordinates
        self.time = find_axis("time", ds) & self.coords
        self.x = find_axis("X", ds) & self.coords
        self.y = find_axis("Y", ds) & self.coords
        self.z = find_axis("Z", ds) & self.coords
        self.z_down = {}  # specify direction of z axis
        for z in self.z:
            self.z_down[z] = None  # None to indicate unknown
            if "positive" in ds[z].ncattrs():
                self.z_down[z] = ds[z].getncattr("positive") == "down"
        self.lon = find_axis("longitude", ds)
        self.lat = find_axis("latitude", ds)
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
            self.static_data = {d for d in self.data if tname not in ds[d].dimensions}
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


if __name__ == "__main__":
    # ds = Dataset("/path/to/your/NetCDF/file")
    ds = Dataset("/path/to/your/NetCDF/file")

    info = DatasetInfo(ds)
    print(f"{info.dims=}")  # all dims
    print(f"{info.coords=}")  # all coordinates
    print(f"{info.coords_with_bounds=}")  # all coords with bounds
    print(f"{info.bounds=}")  # all bound variables
    print(f"{info.bounds_map=}")  # coord: bounds map
    print(f"{info.bounds_dims=}")  # all dims exclusivly used for bounds
    print(f"{info.data=}")  # all available data
    print(f"{info.data_with_all_coords=}")  # all data with only coordinates
    print(f"{info.data_without_coords=}")  # all data without coordinates
    print(f"{info.data_with_aux=}")  # all data with auxiliary coordinates
    print(f"{info.data_dims_map=}")  # data: list(dims) map
    print(f"{info.aux_coords=}")  # all auxiliary coordinates
    print(f"{info.aux_coords_map=}")  # data: aux-coords map
    print(f"{info.x=}")  # all x-axis
    print(f"{info.y=}")  # all y-axis
    print(f"{info.z=}")  # all z-axis
    print(f"{info.z_down=}")  # map for downward pointing z-axis
    print(f"{info.time=}")  # the time axis
    print(f"{info.lat=}")  # all latitude vars
    print(f"{info.lon=}")  # all longitude vars
    print(f"{info.lat_axis=}")  # whether any lat is used as axis
    print(f"{info.lon_axis=}")  # whether any lon is used as axis
    print(f"{info.all_axes=}")  # all axes variables
    print(f"{info.all_static=}")  # whether all data is static
    print(f"{info.static_data=}")  # all static data
    print(f"{info.temporal_data=}")  # all temporal data

    var = "your NetCDF Varaible"
    order = info.get_axes_order(info.data_dims_map[var])
    print(var, order)
    print(f"reversed: {check_order_reversed(order)}")
