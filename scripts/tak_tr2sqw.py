#!/usr/bin/env python3

import numpy as np
import argparse, scipy.fft, time
from Cinema.Tak.Analysor import AnaVDOS, AnaSFactor, AnaSF2VD
import matplotlib.pyplot as plt

#######################################################
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, default='',
                    dest='input', help='input json file')

args = parser.parse_args()
inputfile=args.input

import h5py

f=h5py.File(inputfile, 'r')
tr = f['trj'][()]
f.close()

Q=1e-2
tr = Q*tr
print(tr.shape)

fftSize = tr.shape[2]
a=np.zeros(fftSize, dtype=np.complex128)
b=np.zeros(fftSize)
start = time.time()

temp = scipy.fft.fft(np.exp(-tr*1j), n=fftSize, axis=2, workers=4)

#
# for i in range(1, tr.shape[0], 3):
#     # print(i)
#     for dim in range(3):
#         temp = np.fft.fftshift(np.fft.fft(np.exp(-tr[i,dim,:]*1j), n=fftSize))
#         a += temp
#         b += temp.real*temp.real+temp.imag*temp.imag

end = time.time()
print("cal elapsed = %s" % (end - start))

# plt.loglog(a.real*a.real+a.imag*a.imag, label='a')
plt.semilogy(b, 'o', label='b')
plt.legend()
plt.show()
