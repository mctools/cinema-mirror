#!/usr/bin/env python3

import numpy as np
import argparse, scipy.fft, time
from Cinema.Tak.Analysor import AnaVDOS, AnaSFactor, AnaSF2VD, parFFT, parFFTs
import matplotlib.pyplot as plt
import h5py




#######################################################
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, default='',
                    dest='input', help='input json file')

args = parser.parse_args()
inputfile=args.input
df = DynamicFactor(inputfile)
sq =[]
for i in range(20):
    inco, coh = df.cal(i+1.)
    sq.append(coh.sum())

plt.plot(sq)
plt.show()

# plt.semilogy(inco, label='inco')
# plt.semilogy(coh, label='coh')
# plt.legend()
# plt.show()
