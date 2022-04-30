#!/usr/bin/env python3

import numpy as np
from PiXiu.PhononCalc import MeshCell

kt =0.0253 #temperature in kelvin
calc = MeshCell('./data/Al/mesh.hdf5', './data/Al/cell.json', kt)

enSize=3
QSize=3
maxQ=1.

reference = np.array([[4.6228780021719427e+11, 1.8611751313155701e+11, 1.9983451288198476e+09],
 [1.6770028266259140e+10, 6.5173703935609990e+12, 1.5761567067503064e+12],
 [0.0000000000000000e+00, 3.7111931390436929e+12, 2.8527382358669937e+12]])


# import profile
# profile.run('calc.calcPowder(maxQ, enSize, QSize, jump=1)')
hist = calc.calcPowder(maxQ, enSize, QSize, jump=1)
np.set_printoptions(precision=16)
print(hist.getWeight())

np.testing.assert_array_equal(reference, hist.getWeight())

#calc.show(hist.getWeight(), hist.xedge, hist.yedge)
