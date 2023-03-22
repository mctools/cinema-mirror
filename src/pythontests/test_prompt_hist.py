#!/usr/bin/env python3

from io import BytesIO
import numpy as np
from  Cinema.Prompt.Histogram import Hist1D, Hist2D
import matplotlib.pyplot as plt

xbin=10
xmin=0.
xmax=1.

ybin=100
ymin=0.
ymax=1.

hist1d=Hist1D(xmin, xmax, xbin)
hist2d=Hist2D(xmin, xmax, xbin, ymin, ymax, ybin)

#
# for i in range(100):
#     hist=Hist2D(xmin, xmax, xbin, ymin, ymax, ybin)
#     input = np.random.random([2,1])
#     x=input[0]
#     y=input[1]
#     hist.fill(x,y)
#
#     data=hist.getWeight()
#
#     histLoc = np.nonzero(data)
#     xi = ((x-xmin)/(xmax-xmin)*xbin).astype(int)
#     yi = ((y-ymin)/(ymax-ymin)*ybin).astype(int)
#
#     np.testing.assert_equal(histLoc[0], xi)
#     np.testing.assert_equal(histLoc[1], yi)
#
# print('passed 100 times')
