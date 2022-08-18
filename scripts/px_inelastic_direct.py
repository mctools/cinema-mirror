#!/usr/bin/env python3

from re import T
import numpy as np
from Cinema.PiXiu.PhononCalc import MeshCell, MeshQE
from Cinema.Interface.Utils import findData
import argparse

#parameters
#temperature in kelvin
#upper limit for the Q, maxQ, float
#frequency bin size for the histogram, freSize, int
#Q bin size for the histogram, QSize, int
#stepping for the hkl, int
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--temperature', action='store', type=float,
                    dest='temp', help='temperature in kelvin', required=True)
parser.add_argument('-u', '--upper-limit-Q', action='store', type=float, default=10.0,
                    dest='maxQ', help='upper limit for the Q')
parser.add_argument('-f', '--frequency-bin-size', action='store', type=int, default=200,
                    dest='freSize', help='frequency bin size for the histogram')
parser.add_argument('-q', '--Q-bin-size', action='store', type=int, default=300,
                    dest='QSize', help='Q bin size for the histogram')
parser.add_argument('-s', '--step', action='store', type=int, default=1,
                    dest='step', help='stepping for the hkl')
parser.add_argument('-o', '--output-file-name', action='store', default='qehist.h5',
                    dest='output', help='output file name')
args=parser.parse_args()

temp = args.temp #temperature in kelvin
maxQ = args.maxQ
freSize = args.freSize
QSize = args.QSize
step = args.step
output = args.output

# calc = MeshCell(findData('Al/mesh.hdf5'), findData('Al/cell.json'), kt)
calc = MeshQE('mesh.hdf5', 'out_relax.xml', temp)
# import profile
hist = calc.calcPowder(maxQ, freSize, QSize, step)

hist.save(output)
#save info?

hist.plot()

import matplotlib.pyplot as plt
plt.show()
