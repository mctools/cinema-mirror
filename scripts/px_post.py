#!/usr/bin/env python3

import numpy as np
from Cinema.PiXiu.PhononCalc import MeshCell, MeshQE
from Cinema.Interface.Utils import findData


temp = 77 #temperature in kelvin

# calc = MeshCell(findData('Al/mesh.hdf5'), findData('Al/cell.json'), kt)

calc = MeshQE('mesh.hdf5', 'out_relax.xml', temp)


enSize=100
QSize=200
maxQ=3.

# import profile
# profile.run('calc.calcPowder(maxQ, enSize, QSize, jump=1)')
hist = calc.calcPowder(maxQ, enSize, QSize, jump=1)

hist.plot()

import matplotlib.pyplot as plt
plt.show()
