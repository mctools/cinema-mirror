#!/usr/bin/env python3

import numpy as np
from PiXiu.PhononCalc import Hdf5Mesh
from PiXiu.Common.Units import *

#['lattice', 'mass', 'scl', 'pos']
lattice=np.array([[2.27983857518, 0.0, 0.0],
[-1.13991928759, 1.97439812263, 0.0],
[0.0 ,0.0 ,3.527773]])
mass=np.array([9,9])
pos=np.array([[0.333333333333, 0.666666666667, 0.75],[0.666666666667, 0.333333333333, 0.25]])
bc = np.array([0.779, 0.779	]) #scat length
kt=1002*8.6173324e-5

calc = Hdf5Mesh(lattice, mass, pos, bc, kt, './bigdata/Be_554_888/meshT.hdf5')

# import matplotlib.pyplot as plt
# cen, hist = calc.dos()
# plt.plot(cen, hist)
# plt.show()

print('First qpoint', calc.qpoint[0])
# print('matrix', calc.calmsd2())
# print('isotropic msd', calc.isoMsd())



enSize=250
maxNum=1
jump=1

qSize=200*maxNum
maxQ=5

hist = calc.calcPowder(maxQ, enSize, qSize, jump=1)

calc.show(hist.getWeight(), hist.xedge, hist.yedge)
