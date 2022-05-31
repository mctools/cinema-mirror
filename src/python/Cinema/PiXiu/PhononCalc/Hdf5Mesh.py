import h5py
import numpy as np
from .CalcBase import CalcPowder
from Cinema.Interface.units import *

class Hdf5Mesh(CalcPowder):
    def __init__(self, lattice, mass, pos, bc, kt, fileName):
        hf = h5py.File(fileName,'r')
        en=hf['frequency'][()]*THz*2*np.pi*hbar
        eigv=hf['eigenvector'][()]
        eigvShape=eigv.shape
        nAtom = mass.size
        eigv=eigv.reshape([eigvShape[0], nAtom*3, nAtom, 3]) #fixme: magnitude of each eigv is slight different
        # for aQpoint in eigv:
        #     for aMode in aQpoint:
        #         for anAtom in aMode:
        #             anAtom /= np.linalg.norm(anAtom)

        mesh=hf['mesh'][()]
        qpoint=hf['qpoint'][()]
        weight=hf['weight'][()]/(mesh[0]*mesh[1]*mesh[2])
        hf.close()
        super().__init__(lattice, mass, pos, bc, qpoint, en, eigv, weight, kt)
