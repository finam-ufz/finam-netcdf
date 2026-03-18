from datetime import datetime

import finam as fm
import matplotlib.pyplot as plt
from finam_plot import ImagePlot

from finam_netcdf import NetCdfStaticReader, Variable

path = "tests/data/temp.nc"

reader = NetCdfStaticReader(path, [Variable("tmin", io_name="T", slices={"time": 0})])
viewer = ImagePlot()

composition = fm.Composition([reader, viewer])

reader.outputs["T"] >> viewer.inputs["Grid"]

composition.run()

plt.show(block=True)
