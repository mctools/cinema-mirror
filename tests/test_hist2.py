#!/usr/bin/env python3

from io import BytesIO
import numpy as np
from  PiXiu.Utils.cHist import NumpyHist2D
from PiXiu.Utils.Histogram import Hist2D

import matplotlib.pyplot as plt

xbin=10
xmin=0.
xmax=1.

ybin=100
ymin=0.
ymax=1.

hist=NumpyHist2D(xbin, xmin, xmax, ybin, ymin, ymax)
histSlow=Hist2D(xbin, ybin, [[xmin, xmax], [ymin, ymax]])

for i in range(100):
    data=np.random.random([2, 10000])
    hist.fill(data[0], data[1])
    histSlow.slowfill(data[0], data[1])


np.testing.assert_array_equal(hist.getHistVal(), histSlow.hist)

print('passed')
