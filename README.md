# FINAM-netCDF

FINAM components for reading and writing spatial and temporal data from and to [NetCFD](https://www.unidata.ucar.edu/software/netcdf/) files.
Supports reading and writing regular 2-dimensional raster data from and to multidimensional datasets.

All components use finam's `Grid` type for data exchange.

## Installation

```shell
$ pip install git+https://git.ufz.de/FINAM/finam-netcdf.git
```

## Usage

See also the [examples](examples) for complete coupling scripts using the components described here.

### Readers

The package provides two types of NetCDF reader components:

* `NetCdfInitReader` for reading starting conditions during initialization
* `NetCdfTimeReader` for reading time series of rasters

Both components can read multiple variables from a single dataset.

#### `NetCdfInitReader`

Reads once during initialization of the coupling setup.
All coordinate dimensions except `x` and `y` must be fixed at a certain index.

```python
from finam_netcdf.reader import Layer, NetCdfInitReader

path = "tests/data/lai.nc"
reader = NetCdfInitReader(
    path=path,
    outputs={
        "LAI": Layer(var="lai", x="lon", y="lat", fixed={"time": 0}),
    }
)
```

#### `NetCdfTimeReader`

Reads once on each time step, where time steps are defined by the time dimension provided by the dataset (but see also [Time manipulation](#time-manipulation)).
All coordinate dimensions except `x`, `y` and `time` must be fixed at a certain index.

```python
from finam_netcdf.reader import Layer, NetCdfTimeReader

path = "tests/data/lai.nc"
reader = NetCdfTimeReader(
    path=path, 
    outputs={"LAI": Layer(var="lai", x="lon", y="lat")},
    time_var="time"
)
```

When multiple variables/layers are read, they must all use the same time dimension (i.e. they must have common time steps).

##### Time manipulation

In some cases, it may not be desirable to use time data from a dataset directly.
The example dataset `lai.nc` used above contains 12 LAI rasters along the temporal axis, one for each month of the year.
This example cycles through the 12 rasters every year:

```python
from datetime import datetime
from finam_netcdf.reader import Layer, NetCdfTimeReader

start = datetime(2000, 1, 1)

def to_time_step(tick, _last_time, _last_index):
    year = start.year + tick // 12
    month = 1 + tick % 12
    return datetime(year, month, 1), tick % 12

path = "tests/data/lai.nc"
reader = NetCdfTimeReader(
    path=path, 
    outputs={"LAI": Layer(var="lai", x="lon", y="lat")},
    time_var="time",
    time_callback=to_time_step,
)
```

#### Outputs

Component outputs can be accessed by the keys used for `outputs`, e.g. for linking:

```python
reader.outputs()["LAI"] >> viewer.inputs()["Grid"]
```

### Writers

TODO
