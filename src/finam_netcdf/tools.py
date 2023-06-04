"""NetCDF helper classes and functions"""
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
    Extracts the layer information from a dataset following CF convention
    """
    layers = []
    var_list = []
    time_var = None
    x_var = None
    y_var = None
    z_var = None
    ATTRS = {
        "time": {
            "axis": "T",
            "units": "since",
            "standard_name": "time",
            "long_name": "time",
        },
        "longitude": {
            "axis": "X",
            "units": [
                "degrees_east",
                "degree_east",
                "degree_E",
                "degrees_E",
                "degreeE",
                "degreesE",
            ],
            "standard_name": ["longitude", "lon", "degrees_east", "xc"],
            "long_name": ["longitude", "lon", "degrees_east", "xc"],
        },
        "latitude": {
            "axis": "Y",
            "units": [
                "degrees_north",
                "degree_north",
                "degree_N",
                "degrees_N",
                "degreeN",
                "degreesN",
            ],
            "standard_name": ["latitude", "lat", "degrees_north", "yc"],
            "long_name": ["latitude", "lat", "degrees_north", "yc"],
        },
        "Z": {
            "axis": "Z",
            "standard_name": [
                "level",
                "pressure level",
                "depth",
                "height",
                "vertical level",
                "elevation",
                "altitude",
            ],
            "long_name": [
                "level",
                "pressure level",
                "depth",
                "height",
                "vertical level",
                "elevation",
                "altitude",
            ],
        },
    }

    for var, data in dataset.variables.items():
        if len(data.dimensions) > 4:
            raise ValueError(f"Variable {data.name} has more than 4 possible dimensions (T, Z, Y, X).")
        
        # getting parameter variables
        if len(data.dimensions) > 2:
            var_list.append(data.name)
        else:
            # getting dimensions (T, Z, X, Y)
            if var in dataset.dimensions.keys():
                if "calendar" in data.ncattrs():
                    time_var = _check_var(time_var, data.name, check_var=True)

                elif "positive" in data.ncattrs():
                    z_var = _check_var(z_var, data.name, check_var=True)
                else:
                    time_var, check_var = _check_var_attr(time_var, data, var, ATTRS["time"])
                    time_var = _check_var(time_var, data.name, check_var)

                    z_var, check_var = _check_var_attr(z_var, data, var, ATTRS["Z"])
                    z_var = _check_var(z_var, data.name, check_var)

                    x_var, check_var = _check_var_attr(x_var, data, var, ATTRS["longitude"])
                    x_var = _check_var(x_var, data.name, check_var)

                    y_var, check_var = _check_var_attr(y_var, data, var, ATTRS["latitude"])
                    y_var = _check_var(y_var, data.name, check_var)
                    
    xyz = tuple(v for v in [x_var, y_var] if v is not None)

    if len(xyz) < 2:
        raise ValueError(f"CF conventions not met or coordinates missing. Input (X,Y) NetCDF coordinates: {xyz}.")
        
    for var in var_list:
        if dataset[var].dimensions == 4:
            if z_var == None:
                raise ValueError(f"Input NetCDF coordinate Z does not comply with CF conventions!")
            else:
                layers.append(Layer(var, xyz, fixed={z_var: 0}, static=time_var is None))
        else:
            layers.append(Layer(var, xyz, static=time_var is None))

    return time_var, layers


def _check_var(old, new, check_var):
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


def extract_grid(dataset, layer, time_index=None):
    """Extracts a 2D data array from a dataset

    Parameters
    ----------
    dataset : netCDF4.DataSet
        The input dataset
    layer : Layer
        The layer definition
    time_index : int, optional
        time_index indice of a variable
    """
    var_data = dataset[layer.var]
    # # var_data.dimensions is at least always (time, x ,y) - could be:('time', 'lev', 'lat', 'lon')
    # if len(var_data.dimensions) > 3:
    #     raise ValueError(f"NetCDF variable {layer.var} has more than 3 dimensions")
    # if len(var_data.dimensions) != len(layer.xyz):  # len(layer.xyz) will get just lat and lon, but len(var_data.dimensions) could have more
    #     raise ValueError(
    #         f"NetCDF variable {layer.var} has a different number of dimensions than given axes"
    #     )

    for ax in layer.xyz:
        if ax not in var_data.dimensions:
            raise ValueError(
                f"Dimension {ax} not available for NetCDF variable {layer.var}"
            )

    meta = {}
    for name in var_data.ncattrs():
        meta[name] = getattr(var_data, name)

    # filtering variable data by time_index as np array
    if time_index is None:
        var_data = var_data[:].filled(fill_value=np.nan)
    else:
        var_data = var_data[time_index].filled(fill_value=np.nan)

    # getting coordinates data
    xyz_data = [dataset.variables[ax] for ax in layer.xyz]
    axes = np.array([np.array(ax[:]) for ax in xyz_data], dtype=object)

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

    # creating the time dimension
    times = None
    if not layer.static and "time" in dataset.dimensions:
        nctime = dataset["time"][time_index]
        time_cal = dataset["time"].calendar
        time_unit = dataset.variables["time"].units
        times = num2date(
            nctime, units=time_unit, calendar=time_cal, only_use_cftime_datetimes=False
        )
        times = np.array(times).astype("datetime64[ns]")
        times = times.astype("datetime64[s]").tolist()

    info = fm.Info(time=times, grid=grid, meta=meta)

    return info, fm.UNITS.Quantity(var_data, info.units)


def create_point_axis(cell_axis):
    """Create a point axis from a cell axis"""
    diffs = np.diff(cell_axis)
    mid = cell_axis[:-1] + diffs / 2
    first = cell_axis[0] - diffs[0] / 2
    last = cell_axis[-1] + diffs[-1] / 2
    return np.concatenate(([first], mid, [last]))
