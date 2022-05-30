#!/usr/bin/env python3

import numpy as np
import argparse, scipy.fft, time
from Cinema.Tak.Analysor import DynamicFactor
import matplotlib.pyplot as plt
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, default='',
                    dest='input', help='input json file')

args = parser.parse_args()
inputfile=args.input
df = DynamicFactor(inputfile)

Q=1

# coh = df.calCoherent(Q)
# plt.semilogy(coh, label='coh')

inco = df.calIncoherent(Q)
plt.semilogy(inco, label='inco')

inco = df.calIncoherent(Q, True)
plt.semilogy(inco, label='inco windowed')

plt.legend()
plt.show()

# sq =[]
# for i in range(20):
#     coh = df.calCoherent(i+1.)
#     sq.append(coh.sum())
#
# plt.plot(sq)
# plt.show()
