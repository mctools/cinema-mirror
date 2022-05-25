#!/usr/bin/env python3

import numpy as np
import argparse
from Cinema.Tak.Analysor import AnaVDOS


#######################################################3
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, default='mp-13_Fe.json',
                    dest='input', help='input json file')
parser.add_argument('-c', '--numcpu', action='store', type=int, default=-1,
                    dest='numcpu', help='number of CPU')

args = parser.parse_args()

inputfile=args.input
numcpu=args.numcpu


t = AnaVDOS(inputfile)
vdos = t.vdos(numcpu)
vdos_python = t.vdos_python()

import matplotlib.pyplot as plt
plt.plot(np.abs(vdos), label='C++')
plt.plot(np.abs(vdos_python), label='python')
plt.legend()
plt.show()
