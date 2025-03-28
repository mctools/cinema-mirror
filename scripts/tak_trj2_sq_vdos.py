#!/usr/bin/env python3

import numpy as np
import argparse
from Cinema.Tak.analysor import AnaVDOS, AnaSFactor, AnaSF2VD
import matplotlib.pyplot as plt
from Cinema.Interface import units, plotStyle
plotStyle()
#######################################################
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, default='mp-13_Fe.json',
                    dest='input', help='input json file')
parser.add_argument('-c', '--numcpu', action='store', type=int, default=-1,
                    dest='numcpu', help='number of CPU')
parser.add_argument('-q', '--numq', action='store', type=int, default=0,
                    dest='numq', help='number of Qpoint')
parser.add_argument('--test', action='store_true', dest='test', help='run test')
parser.add_argument('-p','--plot', action='store_true', dest='plot', help='plot figure')
parser.add_argument('-o', '--output', action='store', type=str, default='',
                    dest='output', help='output')
parser.add_argument('--atomoffset',  type=int, nargs='+', dest='atomoffset',
                     help='the atom offset for vdos calculation')

args = parser.parse_args()
inputfile=args.input
numcpu=args.numcpu

# the trajactory the coherent S(Q) does not need to be wrapped
# calculate S(Q)
s = AnaSFactor(inputfile)
if args.numq:
    q, sq = s.getSq(args.numq)
    if args.plot:
        plt.figure()
        plt.plot(q, sq)
        plt.show()

# swap axes from atomid, frameid, pos_dim to atomid, pos_dim, frameid to accumulate 1D incoherent analysis
anavdos = AnaSF2VD(s)

if args.test:
    vdos = anavdos.vdos(0, numcpu)
    vdos_python = anavdos.vdos_python(0)
    np.testing.assert_allclose(vdos_python, vdos[1])
    print('pass!')
    import sys
    sys.exit()

# t, msd = anavdos.msd(1, numcpu)
# plt.figure()
# plt.plot(t/1e-12, msd, label='C++')
# plt.show()


if args.atomoffset:
    for offset in args.atomoffset:
        fre, vdos = anavdos.vdos(offset, numcpu)
        if args.plot:
            plt.figure()
            plt.plot(fre*units.hbar, np.abs(vdos), label='C++')
            plt.show()

if args.output:
    anavdos.saveTrj(args.output)
