"""NetCDF helper classes and functions"""
import copy

import finam as fm
import numpy as np


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
    """Extracts layer information from a dataset"""
    layers = []
    time_var = None

    for var, data in dataset.data_vars.items():
        coords = data.coords
        x_var = None
        y_var = None
        z_var = None
        for co, coord in coords.items():
            if "axis" in coord.attrs:
                ax = coord.attrs["axis"]
                if ax == "T":
                    time_var = _check_var(time_var, co)
                elif ax == "X":
                    x_var = _check_var(x_var, co)
                elif ax == "Y":
                    y_var = _check_var(y_var, co)
                elif ax == "Z":
                    z_var = _check_var(z_var, co)
            else:
                if coord.dtype.type == np.datetime64:
                    time_var = _check_var(time_var, co)

        xyz = tuple(v for v in [x_var, y_var, z_var] if v is not None)
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
    dataset : xarray.DataSet
        The input dataset
    layer : Layer
        The layer definition
    fixed : dict of str, int, optional
        Fixed variables with their indices
    """
    variable = dataset[layer.var].load()
    xyz = [variable.coords[ax] for ax in layer.xyz]

    xdata = variable.isel(layer.fixed if fixed is None else dict(layer.fixed, **fixed))

    if len(xdata.dims) > 3:
        raise ValueError(f"NetCDF variable {layer.var} has more than 3 dimensions")
    if len(xdata.dims) != len(layer.xyz):
        raise ValueError(
            f"NetCDF variable {layer.var} has a different number of dimensions than given axes"
        )

    for ax in layer.xyz:
        if ax not in xdata.dims:
            raise ValueError(
                f"Dimension {ax} not available for NetCDF variable {layer.var}"
            )

    axes = [ax.data.copy() for ax in xyz]

    # re-order axes to xyz
    xdata = xdata.transpose(*layer.xyz)

    # flip to make all axes increasing
    for i, is_increase in enumerate(fm.data.check_axes_monotonicity(axes)):
        if not is_increase:
            ax_name = layer.xyz[i]
            xdata = xdata.reindex(**{ax_name: xdata[ax_name][::-1]}, copy=False)

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

    meta = copy.copy(xdata.attrs)

    # re-insert the time dimension
    time = None
    if not layer.static and "time" in xdata.coords:
        time = fm.data.to_datetime(xdata["time"].data)

    xdata = xdata.drop_vars(xdata.coords)
    xdata = xdata.expand_dims(dim="time", axis=0)

    info = fm.Info(time=time, grid=grid, meta=meta)

    return info, xdata.data * info.units


def create_point_axis(cell_axis):
    """Create a point axis from a cell axis"""
    diffs = np.diff(cell_axis)
    mid = cell_axis[:-1] + diffs / 2
    first = cell_axis[0] - diffs[0] / 2
    last = cell_axis[-1] + diffs[-1] / 2
    return np.concatenate(([first], mid, [last]))
