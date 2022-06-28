#!/usr/bin/env python3

import numpy as np
from Cinema.PiXiu.PhononCalc import MeshCell
from Cinema.Interface.Utils import findData
from Cinema.Interface.units import *

temp =0.0253/boltzmann #temperature in kelvin

calc = MeshCell(findData('Al/mesh.hdf5'), findData('Al/cell.json'), temp)

enSize=3
QSize=3
maxQ=1.

reference = np.array([[4.6228779711143628e+11, 1.8611751206224219e+11, 1.9983451195870922e+09],
 [1.6770028159554180e+10, 6.5173703574206064e+12, 1.5761567004272585e+12],
 [0.0000000000000000e+00, 3.7111931199351382e+12, 2.8527382241319131e+12]])


# import profile
# profile.run('calc.calcPowder(maxQ, enSize, QSize, jump=1)')
hist = calc.calcPowder(maxQ, enSize, QSize, jump=1)
np.set_printoptions(precision=16)
print(hist.getWeight())

np.testing.assert_allclose(reference, hist.getWeight(), rtol=1e-15)

#calc.show(hist.getWeight(), hist.xedge, hist.yedge)
