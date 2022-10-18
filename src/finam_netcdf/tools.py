"""NetCDF helper classes and functions"""
import copy

import finam as fm
import numpy as np


class Layer:
    """
    Defines a NetCDF layer (2D data array).

    :param var: layer variable
    :param x: x coordinate variable
    :param y: y coordinate variable
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
    axes_increase = fm.data.check_axes_monotonicity(axes)

    # re-order axes to xyz
    xdata = xdata.transpose(*layer.xyz)

    # flip to make all axes increasing
    for i, is_increase in enumerate(axes_increase):
        if not is_increase:
            ax_name = layer.xyz[i]
            xdata.reindex(**{ax_name: xdata[ax_name][::-1]}, copy=False)

    # calculate properties of uniform grids
    spacing = fm.data.check_axes_uniformity(axes)
    origin = [ax[0] for ax in axes]
    is_uniform = not any(np.isnan(spacing))

    # note: we use point-associated data here.
    if is_uniform:
        dims = [len(ax) for ax in axes]
        grid = fm.UniformGrid(
            dims,
            axes_names=layer.xyz,
            spacing=tuple(spacing),
            origin=tuple(origin),
            data_location=fm.Location.POINTS,
        )
    else:
        grid = fm.RectilinearGrid(
            axes, axes_names=layer.xyz, data_location=fm.Location.POINTS
        )

    xdata = fm.data.quantify(xdata)

    # re-insert the time dimension
    if fm.data.has_time(xdata):
        xdata = xdata.expand_dims(dim="time", axis=0)

    meta = copy.copy(xdata.attrs)
    info = fm.Info(grid=grid, meta=meta)

    return info, xdata
