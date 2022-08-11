#!/usr/bin/env python3

import numpy as np
from Cinema.PiXiu.PhononCalc import MeshCell, MeshQE
from Cinema.Interface.Utils import findData


#parameters
#temperature in kelvin
#upper limit for the Q, maxQ, float
#frequency bin size for the histogram, freSize, int
#Q bin size for the histogram, QSize, int
#stepping for the hkl, int

temp = 20 #temperature in kelvin

# calc = MeshCell(findData('Al/mesh.hdf5'), findData('Al/cell.json'), kt)
calc = MeshQE('mesh.hdf5', 'out_relax.xml', temp)

freSize=200
QSize=300
maxQ=10

# import profile
hist = calc.calcPowder(maxQ, freSize, QSize, step=1)

hist.save('qehist.h5')
hist.plot()

import matplotlib.pyplot as plt
plt.show()
