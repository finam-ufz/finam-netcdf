from datetime import datetime

import finam as fm
import matplotlib.pyplot as plt
from finam_plot import ContourPlot

from finam_netcdf import NetCdfStaticReader, Variable

path = "tests/data/lai.nc"

reader = NetCdfStaticReader(path, [Variable("lai", io_name="LAI", slices={"time": 0})])
viewer = ContourPlot()

composition = fm.Composition([reader, viewer])

reader.outputs["LAI"] >> viewer.inputs["Grid"]

composition.run()

plt.show(block=True)
