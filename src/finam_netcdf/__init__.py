"""NetCDF file I/O components for FINAM"""

import numpy as np
from finam.data.grid import Grid, GridSpec


class Layer:
    """
    Defines a NetCDF layer (2D data array).

    :param var: layer variable
    :param x: x coordinate variable
    :param y: y coordinate variable
    :param fixed: dictionary for further, fixed index coordinate variables (e.g. 'time')
    """

    def __init__(self, var: str, x: str, y: str, fixed=None):
        self.var = var
        self.x = x
        self.y = y
        self.fixed = fixed or {}


def extract_grid(dataset, layer, fixed=None):
    """Extracts a 2D data array from a dataset"""
    variable = dataset[layer.var].load()
    x = variable.coords[layer.x]
    y = variable.coords[layer.y]

    xmin = x.data.min()
    xmax = x.data.max()
    ymin = y.data.min()
    ymax = y.data.max()

    cellsize_x = (xmax - xmin) / (x.shape[0] - 1)
    cellsize_y = (ymax - ymin) / (y.shape[0] - 1)

    if abs(cellsize_x - cellsize_y) > 1e-8:
        raise ValueError(
            "Only raster data with equal resolution in x and y direction is supported."
        )

    fx = layer.fixed if fixed is None else dict(layer.fixed, **fixed)
    extr = variable.isel(fx)

    if len(extr.dims) != 2:
        raise ValueError(f"NetCDF variable {layer.var} has dimensions != 2")

    if extr.dims[0] == layer.x and extr.dims[1] == layer.y:
        transpose = True
    elif extr.dims[0] == layer.y and extr.dims[1] == layer.x:
        transpose = False
    else:
        raise ValueError(
            f"NetCDF variable {layer.var} dimensions do not include x and y ({layer.x}, {layer.y})"
        )

    x_flip = x.data[0] > x.data[-1]
    y_flip = y.data[0] < y.data[-1]

    arr = extr.data
    if transpose:
        arr = arr.T
    if y_flip:
        arr = np.flipud(arr)
    if x_flip:
        arr = np.fliplr(arr)

    flat = arr.flatten()

    grid: Grid = Grid(
        GridSpec(
            ncols=x.shape[0],
            nrows=y.shape[0],
            cell_size=cellsize_x,
            xll=xmin - 0.5 * cellsize_x,
            yll=ymin - 0.5 * cellsize_x,
        ),
        data=flat,
    )

    return grid
