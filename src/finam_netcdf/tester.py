from datetime import datetime

import finam as fm
import netCDF4 as nc
import numpy as np
import pandas as pd
import tools


def _check_var(old, new):
    if old is None or old == new:
        return new
    raise ValueError(f"Axis already defined as {old}. Found second axis {new}.")


tavg_file = "/Users/install/Documents/UFZ/finam/finam-netcdf/tests/data/temp.nc"
tavg = nc.Dataset(tavg_file)

time_var, layers = tools.extract_layers(tavg)

# print(time_var, layers)

output_vars = {l.var: l for l in layers}

print(output_vars)

# for o, layer in output_vars.items():
#            outputs.add(name=o, static=layer.static)


# =========================================================================
layers = []
var_list = []
time_var = None
x_var = None
y_var = None
z_var = None

for var, data in tavg.variables.items():
    # getting measured variables with at least time,lon,lat info
    # print(var)
    if len(data.dimensions) > 2:
        var_list.append(data.name)
    # assigning axis
    else:
        if "axis" in data.ncattrs():
            ax = data.axis
            if ax == "T":
                time_var = _check_var(time_var, data.name)
            elif ax == "X":
                x_var = _check_var(x_var, data.name)
            elif ax == "Y":
                y_var = _check_var(y_var, data.name)
            elif ax == "Z":
                z_var = _check_var(z_var, data.name)
        else:
            if "calendar" in data.ncattrs():
                time_var = _check_var(time_var, data.name)

# print(var_list, x_var, y_var, z_var, time_var)
# =========================================================================
