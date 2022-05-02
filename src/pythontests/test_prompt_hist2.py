#!/usr/bin/env python3

from io import BytesIO
import numpy as np
from  Cinema.Prompt.Math.Hist import Hist2D, NumpyHist2D

import matplotlib.pyplot as plt

xbin=10
xmin=0.
xmax=1.

ybin=100
ymin=0.
ymax=1.

hist=NumpyHist2D(xbin, ybin, [[xmin, xmax], [ymin, ymax]])
histcpp=Hist2D(xmin, xmax, xbin, ymin, ymax, ybin)

for i in range(100):
    data=np.random.random([2, 10000])
    hist.fill(data[0], data[1])
    histcpp.fillmany(data[0], data[1])


np.testing.assert_array_equal(hist.hist, histcpp.getWeight())

print('passed')
