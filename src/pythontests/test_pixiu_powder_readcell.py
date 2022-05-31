#!/usr/bin/env python3

import numpy as np
from Cinema.PiXiu.PhononCalc import MeshCell
from Cinema.Interface.Utils import findData
kt =0.0253 #temperature in kelvin
#fixme: finddata pacakge
calc = MeshCell(findData('Al/mesh.hdf5'), findData('Al/cell.json'), kt)

enSize=3
QSize=3
maxQ=1.

reference = np.array([[4.6228779711143622e+11, 1.8611751206224219e+11, 1.9983451195870922e+09],
 [1.6770028159554176e+10, 6.5173703574206064e+12, 1.5761567004272585e+12],
 [0.0000000000000000e+00, 3.7111931199351377e+12, 2.8527382241319131e+12]])


# import profile
# profile.run('calc.calcPowder(maxQ, enSize, QSize, jump=1)')
hist = calc.calcPowder(maxQ, enSize, QSize, jump=1)
np.set_printoptions(precision=16)
print(hist.getWeight())

np.testing.assert_array_equal(reference, hist.getWeight())

#calc.show(hist.getWeight(), hist.xedge, hist.yedge)
