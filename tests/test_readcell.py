#!/usr/bin/env python3

import numpy as np
from PiXiu.PhononCalc import MeshCell

kt =0.0253 #temperature in kelvin
calc = MeshCell('./data/Al/mesh.hdf5', './data/Al/cell.json', kt)

enSize=100
QSize=300
maxQ=5.

# import profile
# profile.run('calc.calcPowder(maxQ, enSize, QSize, jump=1)')
hist = calc.calcPowder(maxQ, enSize, QSize, jump=1)

calc.show(hist.getHistVal(), hist.getXedges(), hist.getYedges())
