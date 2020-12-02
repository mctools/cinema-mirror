#!/usr/bin/env python3

from io import BytesIO
import numpy as np
from  PiXiu.Utils.cHist import NumpyHist2D
import matplotlib.pyplot as plt

xbin=10
xmin=0.
xmax=1.

ybin=100
ymin=0.
ymax=1.


for i in range(1000):
    hist=NumpyHist2D(xbin, xmin, xmax, ybin, ymin, ymax)
    input = np.random.random([2,1])
    x=input[0]
    y=input[1]
    hist.fill(x,y)
    # data=hist.getHistVal()

    hist.save('test.npy')
    with open('test.npy', mode='rb') as file: # b is important -> binary
      fileContent = file.read()
    data=np.load(BytesIO(fileContent))

    histLoc = np.nonzero(data)
    xi = ((x-xmin)/(xmax-xmin)*xbin).astype(np.int)
    yi = ((y-ymin)/(ymax-ymin)*ybin).astype(np.int)

    np.testing.assert_equal(histLoc[0], xi)
    np.testing.assert_equal(histLoc[1], yi)

print('passed 1000 times')
