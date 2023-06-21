import finam as fm
import netCDF4 as nc
import numpy as np
from tools import Layer

temp_file = "/Users/install/Documents/UFZ/finam/finam-netcdf/tests/data/temp.nc"
lai_file = "/Users/install/Documents/UFZ/finam/finam-netcdf/tests/data/lai.nc"
tavg_file = "/Users/install/Documents/UFZ/finam/finam-netcdf/tests/data/tavg.nc"

# nc files from mHM
eabs_file = "/Users/install/Documents/UFZ/mhm/test_domain/input/meteo/eabs.nc"
mask_file = "/Users/install/Documents/UFZ/mhm/test_domain/input/meteo/mask.nc"
net_rad_file = "/Users/install/Documents/UFZ/mhm/test_domain/input/meteo/net_rad.nc"
ssrd_file = "/Users/install/Documents/UFZ/mhm/test_domain/input/meteo/ssrd.nc"
strd_file = "/Users/install/Documents/UFZ/mhm/test_domain/input/meteo/strd.nc"
tmax_file = "/Users/install/Documents/UFZ/mhm/test_domain/input/meteo/tmax.nc"
tmin_file = "/Users/install/Documents/UFZ/mhm/test_domain/input/meteo/tmin.nc"
windspeed_file = "/Users/install/Documents/UFZ/mhm/test_domain/input/meteo/windspeed.nc"
pre_file = "/Users/install/Documents/UFZ/mhm/test_domain/input/meteo/pre/pre.nc"
pet_file = "/Users/install/Documents/UFZ/mhm/test_domain/input/meteo/pet/pet.nc"

wgs84_file = "/Users/install/Downloads/test_WGS84.nc"


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


def extract_layers(dataset):
    """Extracts the layer information from a dataset following CF convention"""
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
            raise ValueError(
                f"Variable {data.name} has more than 4 possible dimensions (T, Z, Y, X)."
            )

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
                    time_var, check_var = _check_var_attr(
                        time_var, data, var, ATTRS["time"]
                    )
                    time_var = _check_var(time_var, data.name, check_var)

                    z_var, check_var = _check_var_attr(z_var, data, var, ATTRS["Z"])
                    z_var = _check_var(z_var, data.name, check_var)

                    x_var, check_var = _check_var_attr(
                        x_var, data, var, ATTRS["longitude"]
                    )
                    x_var = _check_var(x_var, data.name, check_var)

                    y_var, check_var = _check_var_attr(
                        y_var, data, var, ATTRS["latitude"]
                    )
                    y_var = _check_var(y_var, data.name, check_var)

    xyz = tuple(v for v in [x_var, y_var] if v is not None)

    # does nc file follow CF conventions?
    if len(xyz) < 2:
        # time & Z can be None
        raise ValueError(
            f"CF conventions not met or coordinates missing. Input (X,Y) NetCDF coordinates: {xyz}."
        )

    # appending to layers
    for var in var_list:
        if dataset[var].dimensions == 4:
            if z_var == None:
                raise ValueError(
                    f"Input NetCDF coordinate Z does not comply with CF conventions!"
                )
            else:
                pass
                layers.append(
                    Layer(var, xyz, fixed={z_var: 0}, static=time_var is None)
                )
        else:
            pass
            print(layers.append(Layer(var, xyz, static=time_var is None)))


file = nc.Dataset(temp_file)
RES = extract_layers(file)
