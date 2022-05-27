#!/usr/bin/env python3

import numpy as np
import argparse
from Cinema.Tak.Analysor import AnaVDOS, AnaSFactor, AnaSF2VD
import matplotlib.pyplot as plt

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

s = AnaSFactor(inputfile)
if args.numq:
    q, sq = s.getSq(args.numq)
    plt.figure()
    plt.plot(q, sq)

anavdos = AnaSF2VD(s)

if args.test:
    vdos = anavdos.vdos(1, numcpu)
    vdos_python = anavdos.vdos_python(1)
    np.testing.assert_allclose(vdos_python, vdos)
    print('pass!')
    import sys
    sys.exit()

if args.atomoffset:
    for offset in args.atomoffset:
        vdos = anavdos.vdos(offset, numcpu)
        if args.plot:
            plt.figure()
            plt.plot(np.abs(vdos), label='C++')
            plt.show()

if args.output:
    anavdos.saveTrj(args.output)
