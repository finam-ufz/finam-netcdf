:html_theme.sidebar_secondary.remove: true

============
FINAM NetCDF
============

NetCDF reader and writer components for the `FINAM <https://finam.pages.ufz.de/>`_ model coupling framework.

Uses :mod:`xarray` for all input and output functionality.

Quickstart
----------

Installation:

.. code-block:: bash

    pip install git+https://git.ufz.de/FINAM/finam-netcdf.git

For available components, see the :doc:`api/index`.

Usage
-----

See the `example scripts <https://git.ufz.de/FINAM/finam-netcdf/-/tree/main/examples>`_
in the GitLab repository for fully functional usage examples.

Readers
^^^^^^^

The package provides two types of NetCDF reader components:

* :class:`.NetCdfStaticReader` for reading starting conditions during initialization
* :class:`.NetCdfReader` for reading time series of rasters

Both components can read multiple variables from a single dataset.

:class:`.NetCdfStaticReader`
""""""""""""""""""""""""""""

Reads once during initialization of the coupling setup.
All coordinate dimensions except those in `xyz` must be fixed at a certain index.

.. testcode:: NetCdfStaticReader

    from finam_netcdf import Layer, NetCdfStaticReader

    path = "tests/data/lai.nc"
    reader = NetCdfStaticReader(
        path=path,
        outputs={
            "LAI": Layer(var="lai", xyz=("lon", "lat"), fixed={"time": 0}),
        }
    )

:class:`.NetCdfReader`
""""""""""""""""""""""

Reads once on each time step, where time steps are defined by the time dimension provided by the dataset (but see also [Time manipulation](#time-manipulation)).
All coordinate dimensions except those in `xyz` and `time` must be fixed at a certain index.

.. testcode:: NetCdfReader

    from finam_netcdf import Layer, NetCdfReader

    path = "tests/data/lai.nc"
    reader = NetCdfReader(
        path=path,
        outputs={"LAI": Layer(var="lai", xyz=("lon", "lat"))},
        time_var="time"
    )

When multiple variables/layers are read, they must all use the same time dimension (i.e. they must have common time steps).

Time manipulation
"""""""""""""""""

In some cases, it may not be desirable to use time data from a dataset directly.
The example dataset `lai.nc` used above contains 12 LAI rasters along the temporal axis, one for each month of the year.
This example cycles through the 12 rasters every year:

.. testcode:: time-manipulation

    from datetime import datetime
    from finam_netcdf import Layer, NetCdfReader

    start = datetime(2000, 1, 1)

    def to_time_step(tick, _last_time, _last_index):
        year = start.year + tick // 12
        month = 1 + tick % 12
        return datetime(year, month, 1), tick % 12


    path = "tests/data/lai.nc"
    reader = NetCdfReader(
        path=path,
        outputs={"LAI": Layer(var="lai", xyz=("lon", "lat"))},
        time_var="time",
        time_callback=to_time_step,
    )

Outputs
"""""""

Component outputs can be accessed by the keys used for `outputs`, e.g. for linking:

.. code-block:: Python

    reader.outputs["LAI"] >> viewer.inputs["Grid"]

Writers
^^^^^^^

The package provides two types of NetCDF writer components:

* :class:`.NetCdfTimedWriter` for writing in predefined, fixed time intervals
* :class:`.NetCdfPushWriter` for writing whenever new data becomes available

Both components can write multiple variables to a single dataset.

:class:`.NetCdfTimedWriter`
"""""""""""""""""""""""""""

Writes time slices regularly, irrespective of input time steps.

.. testcode:: NetCdfTimedWriter

    from datetime import datetime, timedelta
    from finam_netcdf import Layer, NetCdfTimedWriter

    path = "tests/data/out.nc"
    reader = NetCdfTimedWriter(
        path=path,
        inputs={"LAI": Layer(var="lai", xyz=("lon", "lat"))},
        time_var="time",
        start=datetime(2000, 1, 1),
        step=timedelta(days=1),
    )

:class:`.NetCdfPushWriter`
""""""""""""""""""""""""""

Writes time slices as soon as new data becomes available to the inputs.
Note that all input data sources must have the same time step!

.. testcode:: NetCdfPushWriter

    from finam_netcdf import Layer, NetCdfPushWriter

    path = "tests/data/out.nc"
    reader = NetCdfPushWriter(
        path=path,
        inputs={"LAI": Layer(var="lai", xyz=("lon", "lat"))},
        time_var="time"
    )

API References
--------------

Information about the API of FINAM-NetCDF.

.. toctree::
    :hidden:
    :maxdepth: 1

    self

.. toctree::
    :maxdepth: 1

    api/index
