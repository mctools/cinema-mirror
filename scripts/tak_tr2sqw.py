#!/usr/bin/env python3

import numpy as np
import argparse, scipy.fft, time
from Cinema.Tak.analysor import DynamicFactor
import matplotlib.pyplot as plt
import h5py
from Cinema.Interface import units

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, default='',
                    dest='input', help='input json file')

args = parser.parse_args()
inputfile=args.input
df = DynamicFactor(inputfile)

Q=10
plt.figure(figsize=(8, 6))

# fre, coh = df.calCoherent(Q, True)
# plt.semilogy(fre*units.hbar, coh, label='coh')

fre, inco = df.calIncoherent(Q)
plt.semilogy(fre*units.hbar, inco/inco.max(), label='S(Q, $\omega$)')

fre, inco = df.calIncoherent(Q, True)
plt.semilogy(fre*units.hbar, inco/inco.max(), label='S(Q, $\omega$) windowed')

plt.legend()
plt.show()

# sq =[]
# for i in range(20):
#     coh = df.calCoherent(i+1.)
#     sq.append(coh.sum())
#
# plt.plot(sq)
# plt.show()
