#!/usr/bin/env python3

import numpy as np
from Cinema.PiXiu.PhononCalc import MeshCell
from Cinema.Interface.Utils import findData
from Cinema.Interface.units import *
from Cinema.Interface.units import hbar

temp =0.0253/boltzmann #temperature in kelvin

calc = MeshCell(findData('Al/mesh.hdf5'), findData('Al/cell.json'), temp)

enSize=3
QSize=3
maxQ=1.

reference = np.array([[1.3153346518267138e-06, 1.2250477183732239e-04, 3.0428335561124263e-04],
 [1.0374451862495118e-03, 4.2898110971316570e-03, 1.1038233053021969e-05],
 [1.8777064091745220e-03, 2.4427516860952504e-03, 0.0000000000000000e+00]])


# import profile
# profile.run('calc.calcPowder(maxQ, enSize, QSize, step=1)')
hist = calc.calcPowder(maxQ, enSize, QSize, step=1)
np.set_printoptions(precision=16)
print(hist.getWeight())

np.testing.assert_allclose(reference, hist.getWeight(), rtol=1e-15)

#calc.show(hist.getWeight(), hist.xedge, hist.yedge)
