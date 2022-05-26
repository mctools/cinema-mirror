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
parser.add_argument('-q', '--numq', action='store', type=int, default=200,
                    dest='numq', help='number of Qpoint')
parser.add_argument('--test', action='store_true', dest='test', help='run test')
parser.add_argument('-p','--plot', action='store_true', dest='plot', help='plot figure')
parser.add_argument('-o', '--output', action='store', type=str, default='',
                    dest='output', help='output')

args = parser.parse_args()
inputfile=args.input
numcpu=args.numcpu

s = AnaSFactor(inputfile)
q, sq = s.getSq(args.numq)
plt.figure()
plt.plot(q, sq)


anavdos = AnaSF2VD(s)
vdos = anavdos.vdos(numcpu)
if args.plot:
    plt.figure()
    plt.plot(np.abs(vdos), label='C++')
    plt.show()

if args.test:
    vdos_python = t.vdos_python()
    plt.figure()
    plt.plot(np.abs(vdos), label='C++')
    plt.plot(np.abs(vdos_python), label='python')
    plt.legend()
    plt.show()

if args.output:
    t.saveTrj(args.output)
