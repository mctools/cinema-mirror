#!/usr/bin/env python3

from numba import jit, objmode, prange
import numpy as np
import time, h5py
import argparse
from Cinema.Interface import *


_correlation = importFunc('correlation', type_voidp, [type_npdbl2d, type_npdbl1d, type_sizet, type_sizet, type_sizet, type_sizet, type_sizet, type_sizet] )

#######################################################3
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, default='mp-13_Fe.json',
                    dest='input', help='input json file')
args = parser.parse_args()

inputfile=args.input

@jit(nopython=True, fastmath=True, inline='always', cache=True)
def corrFirstFrame(trj, L, nAtom):
    for iA in prange(nAtom):
        for idim in range(3):
            if trj[iA, idim] < 0.:
                trj[iA, idim] += L[idim]
            elif trj[iA, idim] > L[idim]:
                trj[iA, idim] -= L[idim]
            if trj[iA, idim] < 0 or trj[iA, idim] > L[idim]:
                raise RuntimeError('Corrected postion is still outside the box')


@jit(nopython=True, fastmath=True, inline='always', cache=True)
def corr(trj, L, nAtom, trjold):
    halfL = 0.5*L
    i_L = 1./L
    for iA in prange(nAtom):
        for idim in range(3):
            diff = trj[iA, idim]-trjold[iA, idim]
            if np.abs(diff) > halfL[idim]:
                 trj[iA, idim] -= round(diff*i_L[idim])*L[idim]
            if np.abs(trj[iA, idim]-trjold[iA, idim]) > halfL[idim]:
                raise RuntimeError('Correction wrong')

@jit(nopython=True, fastmath=True, cache=True)
def vdosfft(atomictrj, fftsize, atomoffset, totAtom):
    b=np.zeros(fftsize)
    for x in prange(atomoffset, totAtom, 3):
        for i in range(3):
            h = atomictrj[x,i]
            with objmode(a='complex128[:]'):
                hdiff = np.diff(h)
                a = np.fft.fft(hdiff, n=fftsize)
            b += np.abs(a*a.conjugate())
    return b


@jit(nopython=True, fastmath=True, inline='always', cache=True)
def diff(arr):
    sz = arr.size
    res = np.zeros(sz-1)
    for i in prange(sz-1):
        res[i] = arr[i+1]-arr[i]
    return res

# @jit(nopython=True, parallel=True, nogil=True)
def trjdiff(atomictrj, atomoffset, atomPerMolecule):
    if atomoffset > atomPerMolecule:
        raise RuntimeError('atomoffset > atomPerMolecule')
    totframe = atomictrj.shape[2]
    totAtom = atomictrj.shape[0]
    loopSize =  totAtom//atomPerMolecule
    fftsize = totframe*2
    vdos = np.zeros(fftsize, dtype=np.complex128)
    res = np.zeros((loopSize*3, totframe-1))
    idx = 0
    for iAtom in range(atomoffset, totAtom, atomPerMolecule):
        for iDim in range(3):
            res[idx, :] = diff(atomictrj[iAtom, iDim])
            idx += 1
    return res

class Hdf5Trj():
    def __init__(self, inputfile):
        hf = h5py.File(inputfile, 'r')
        self.species = hf["particles/all/species/value"][()]
        self.nAtom = self.species.shape[1]
        self.nFrame = self.species.shape[0]
        print(self.nAtom, self.nFrame)

        self.trj = hf["particles/all/position/value"][()]
        self.box = hf["particles/all/box/edges/value"][()]
        self.time = hf["particles/all/species/time"][()]
        print(self.trj.shape)
        print(self.box.shape)
        print(self.time.shape)
        hf.close()

        start = time.time()
        self.unwrap()
        end = time.time()
        print("unwrap elapsed = %s" % (end - start))

        #swap axes from frameid, atomid, pos_dim to atomid, frameid, pos_dim
        self.atomictrj = np.swapaxes(self.trj, 0, 1)
        del self.trj
        #swap axes from atomid, frameid, pos_dim to atomid, pos_dim, frameid
        self.atomictrj = np.swapaxes(self.atomictrj, 1, 2)

    def unwrap(self):
        #find atoms outside the box in the first frame
        corrFirstFrame(self.trj[0], self.box[0], self.nAtom)
        #unwrap the rest
        for i in range(1, self.nFrame):
            corr(self.trj[i], self.box[i], self.nAtom, self.trj[i-1])

    def vdos(self):
        fftsize = self.nFrame*2
        atomoffset = 1
        totAtom = self.nAtom
        atomictrj = self.atomictrj
        atomPerMolecule=3

        start = time.time()
        diff = trjdiff(atomictrj,atomoffset,atomPerMolecule)
        end = time.time()
        print("vdos elapsed = %s" % (end - start))

        vdos = np.zeros(fftsize)
        _correlation(diff, vdos, 0, diff.shape[0], 1, diff.shape[1], fftsize, 8)

        import matplotlib.pyplot as plt
        plt.plot(np.abs(vdos), label='C++')

        start = time.time()
        vdos = np.zeros(fftsize)
        for i in range(diff.shape[0]):
            temp = np.fft.fft(diff[i], n=fftsize)
            vdos += np.abs(temp)**2

        print(f' diff.shape[0] { diff.shape[0]}')
        end = time.time()
        print("fft elapsed = %s" % (end - start))

        plt.plot(np.abs(vdos),'o', label='python')
        plt.legend()

        plt.show()

t = Hdf5Trj(inputfile)
t.vdos()
