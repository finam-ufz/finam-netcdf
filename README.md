# FINAM-netCDF

FINAM components for reading and writing spatial and temporal data from and to [NetCFD](https://www.unidata.ucar.edu/software/netcdf/) files.
Supports reading and writing regular 2-dimensional raster data from and to multidimensional datasets.

## Installation

```shell
$ pip install git+https://git.ufz.de/FINAM/finam-netcdf.git
```

## Readers

The package provides two types of NetCDF reader components:

* `NetCdfInitReader` for reading starting conditions during initialization
* `NetCdfTimeReader` for reading time series of rasters

Both components can read multiple variables from a single dataset.

**`NetCdfInitReader`** reads once during initialization of the coupling setup.
All coordinate dimensions except `x` and `y` must be fixed at a certain index.

```python
path = "tests/data/lai.nc"
reader = NetCdfInitReader(
    path=path,
    outputs={
        "LAI": Layer(var="lai", x="lon", y="lat", fixed={"time": 0}),
    }
)
```

**`NetCdfTimeReader`** reads once on each time step, where time steps are defined by the time dimension provided by the dataset.
All coordinate dimensions except `x`, `y` and `time` must be fixed at a certain index.

```python
path = "tests/data/lai.nc"
reader = NetCdfTimeReader(
    path=path, 
    outputs={"LAI": Layer(var="lai", x="lon", y="lat")},
    time_var="time"
)
```

When multiple variables/layers are read, they must all use the same time dimension (i.e. they must have common time steps).

Component outputs can be accessed by the keys used for `outputs`, e.g. for linking:

```python
reader.outputs()["LAI"] >> viewer.inputs()["Grid"]
```

## Writers

TODO
