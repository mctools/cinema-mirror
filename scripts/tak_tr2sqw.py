#!/usr/bin/env python3

import numpy as np
import argparse, scipy.fft, time
from Cinema.Tak.Analysor import AnaVDOS, AnaSFactor, AnaSF2VD, parFFT, parFFTs
import matplotlib.pyplot as plt

#######################################################
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, default='',
                    dest='input', help='input json file')

args = parser.parse_args()
inputfile=args.input

import h5py

f=h5py.File(inputfile, 'r')
tr = f['reduced_trj'][()]
f.close()

Q=5
tr = Q*tr
print(tr.shape)

fftSize = tr.shape[2]
a=np.zeros(fftSize, dtype=np.complex128)
b=np.zeros(fftSize)

from numba import jit
@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def scaleQ(exponent, factor):
    return np.exp(-exponent*1j)*factor

@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def incoherent(b):
    return (b.real*b.real + b.imag*b.imag).sum(axis=0)

@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def coherent(b):
    return b.real.sum(axis=0)**2 + b.imag.sum(axis=0)**2

start = time.time()

# (1536, 3, 25001)
# inp = np.exp(-tr[0::3,1,:]*1j) #O
inp = scaleQ(tr, Q) #H

inp = inp.reshape(-1, inp.shape[2])
print(inp.shape)
b = parFFT(inp)

###########################################
inco = incoherent(b)
inco = np.fft.fftshift(inco)

coh = coherent(b)
coh = np.fft.fftshift(coh)

#############################################


end = time.time()
print("cal elapsed = %s" % (end - start))

# plt.loglog(a.real*a.real+a.imag*a.imag, label='a')
plt.semilogy(inco, label='inco')
plt.semilogy(coh, label='coh')

plt.legend()
plt.show()
