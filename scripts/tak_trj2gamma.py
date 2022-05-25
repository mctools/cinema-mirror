#!/usr/bin/env python3

from numba import jit, objmode, prange
import numpy as np
import time, h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, default='mp-13_Fe.json',
                    dest='input', help='input json file')
args = parser.parse_args()

inputfile=args.input

@jit(nopython=True, fastmath=True)
def corr(trj, L, nAtom, trjold=None):
    if trjold is None:
        for iA in prange(nAtom):
            for idim in range(3):
                if trj[iA, idim] < 0.:
                    trj[iA, idim] += L[idim]
                elif trj[iA, idim] > L[idim]:
                    trj[iA, idim] -= L[idim]
                if trj[iA, idim] < 0 or trj[iA, idim] > L[idim]:
                    raise RuntimeError('Corrected postion is still outside the box')
    else:
        halfL = 0.5*L
        i_L = 1./L
        for iA in prange(nAtom):
            for idim in range(3):
                diff = trj[iA, idim]-trjold[iA, idim]
                if np.abs(diff) > halfL[idim]:
                     trj[iA, idim] -= round(diff*i_L[idim])*L[idim]
                if np.abs(trj[iA, idim]-trjold[iA, idim]) > halfL[idim]:
                    raise RuntimeError('Correction wrong')

@jit(nopython=True, fastmath=True)
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

@jit(nopython=True, fastmath=True)
def trjdiff(atomictrj, atomoffset, atomPerMolecule):
    if atomoffset > atomPerMolecule:
        raise RuntimeError('atomoffset > atomPerMolecule')
    totframe = atomictrj.shape[2]
    totAtom = atomictrj.shape[0]
    loopSize =  totAtom//atomPerMolecule
    print(f'loopSize {loopSize}')
    res = np.zeros((loopSize*3, totframe-1))
    idx=0
    for iAtom in prange(atomoffset, totAtom, atomPerMolecule):
        for iDim in range(3):
            a = np.ascontiguousarray(atomictrj[iAtom, iDim])
            res[idx, :] = np.diff(a)
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

        #swap axes from frameid, atomid, pos to atomid, frameid, pos
        self.atomictrj = np.swapaxes(self.trj, 0, 1)
        del self.trj
        #swap axes from atomid, frameid, pos to atomid, pos, frameid
        self.atomictrj = np.swapaxes(self.atomictrj, 1, 2)

    def unwrap(self):
        #find atoms outside the box in the first frame
        corr(self.trj[0], self.box[0], self.nAtom)

        #unwrap the rest
        for i in range(1, self.nFrame):
            corr(self.trj[i], self.box[i], self.nAtom, self.trj[i-1])

    def vdos(self):
        fftsize = self.nFrame
        atomoffset = 1
        totAtom = self.nAtom
        atomictrj = self.atomictrj
        atomPerMolecule=3
        start = time.time()
        # b = vdosfft(atomictrj, fftsize, atomoffset, totAtom)
        diff = trjdiff(atomictrj,atomoffset,atomPerMolecule)
        end = time.time()
        print("vdos elapsed = %s" % (end - start))

        # import matplotlib.pyplot as plt
        # plt.plot(np.abs(b))
        # plt.show()

t = Hdf5Trj(inputfile)
t.vdos()
