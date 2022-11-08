"""NetCDF helper classes and functions"""
import copy

import finam as fm
import numpy as np


class Layer:
    """
    Defines a NetCDF layer (2D data array).

    :param var: layer variable
    :param xyz: coordinate variables in xyz order
    :param fixed: dictionary for further, fixed index coordinate variables (e.g. 'time')
    """

    def __init__(self, var: str, xyz=("x", "y"), fixed=None):
        self.var = var
        self.xyz = xyz
        self.fixed = fixed or {}


def extract_grid(dataset, layer, fixed=None):
    """Extracts a 2D data array from a dataset"""
    variable = dataset[layer.var].load()
    xyz = [variable.coords[ax] for ax in layer.xyz]

    fx = layer.fixed if fixed is None else dict(layer.fixed, **fixed)
    xdata = variable.isel(fx)

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

    axes = [ax.data for ax in xyz]

    # re-order axes to xyz
    xdata = xdata.transpose(*layer.xyz)

    # flip to make all axes increasing
    for i, is_increase in enumerate(fm.data.check_axes_monotonicity(axes)):
        if not is_increase:
            ax_name = layer.xyz[i]
            xdata.reindex(**{ax_name: xdata[ax_name][::-1]}, copy=False)

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

    xdata = fm.data.quantify(xdata)

    # re-insert the time dimension
    time = None
    if fm.data.has_time(xdata):
        xdata = xdata.expand_dims(dim="time", axis=0)
        time = fm.data.get_time(xdata)[0]

    meta = copy.copy(xdata.attrs)
    info = fm.Info(time=time, grid=grid, meta=meta)

    return info, xdata


def create_point_axis(cell_axis):
    """Create a point axis from a cell axis"""
    diffs = np.diff(cell_axis)
    mid = cell_axis[:-1] + diffs / 2
    first = cell_axis[0] - diffs[0] / 2
    last = cell_axis[-1] + diffs[-1] / 2
    return np.concatenate(([first], mid, [last]))
