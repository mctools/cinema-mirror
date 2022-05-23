#!/usr/bin/env python3

from numba import jit
import numpy as np
import time, h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, default='mp-13_Fe.json',
                    dest='input', help='input json file')
args = parser.parse_args()

inputfile=args.input

@jit(nopython=True)
def corr(trj, L, nAtom, trjold=None):
    if trjold is None:
        for iA in range(nAtom):
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
        for iA in range(nAtom):
            for idim in range(3):
                diff = trj[iA, idim]-trjold[iA, idim]
                if np.abs(diff) > halfL[idim]:
                     trj[iA, idim] -= round(diff*i_L[idim])*L[idim]
                if np.abs(trj[iA, idim]-trjold[iA, idim]) > halfL[idim]:
                    raise RuntimeError('Correction wrong')

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

        # start = time.time()
        # self.unwrap()
        # end = time.time()
        # print("Elapsed (after compilation) = %s" % (end - start))

        start = time.time()
        self.unwrap()
        end = time.time()
        print("Elapsed (after compilation) = %s" % (end - start))

    def unwrap(self):
        #find atoms outside the box in the first frame
        corr(self.trj[0], self.box[0], self.nAtom)

        #unwrap the rest
        for i in range(1, self.nFrame):
            corr(self.trj[i], self.box[i], self.nAtom, self.trj[i-1])






Hdf5Trj(inputfile)
