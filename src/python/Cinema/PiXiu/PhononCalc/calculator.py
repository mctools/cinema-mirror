import h5py
import numpy as np
from .calcBase import CalcPowder, CalcBand
from Cinema.Interface.units import *

class Hdf5Powder(CalcPowder):
    def __init__(self, lattice, mass, pos, bc, temperature, fileName, phonIdx):
        hf = h5py.File(fileName,'r')
        en=hf['frequency'][phonIdx]*THz*2*np.pi*hbar
        eigv=hf['eigenvector'][phonIdx]
        eigvShape=eigv.shape
        nAtom = mass.size
        if eigv.ndim != 3:
            raise RuntimeError('eigv.ndim != 3')
        print(f'Hdf5Power eigv shape {eigv.shape}')
        #num of Q points, num of modes, number of atom, vector of size three
        eigv=eigv.reshape([eigvShape[0], nAtom*3, nAtom, 3]) #fixme: magnitude of each eigv is slight different

        mesh=hf['mesh'][()]
        qpoint=hf['qpoint'][phonIdx]
        weight=hf['weight'][phonIdx]/(mesh[0]*mesh[1]*mesh[2])
        hf.close()
        super().__init__(lattice, mass, pos, bc, qpoint, en, eigv, weight, temperature)


class Hdf5Band(CalcBand):
    def __init__(self, lattice, mass, pos, bc, temperature, fileName):
        hf = h5py.File(fileName,'r')
        en=hf['frequency'][()]*THz*2*np.pi*hbar
        eigv=hf['eigenvector'][()]
        eigvShape=eigv.shape
        nAtom = mass.size
        if eigv.ndim != 4:
            raise RuntimeError('eigv.ndim != 4')
        print(f'Hdf5Band eigv shape {eigv.shape}')
        #num of Q points, num of modes, number of atom, vector of size three
        eigv=eigv.reshape([eigvShape[0], eigvShape[0], nAtom*3, nAtom, 3]) #fixme: magnitude of each eigv is slight different

        mesh=hf['mesh'][()]
        qpoint=hf['qpoint'][()]
        weight=hf['weight'][()]
        hf.close()
        super().__init__(lattice, mass, pos, bc, qpoint, en, eigv, weight, temperature)
