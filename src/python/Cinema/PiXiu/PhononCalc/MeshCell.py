import h5py
import numpy as np
import json

from .Hdf5Mesh import Hdf5Mesh
from ..AtomInfo import getAtomMassBC

class MeshCell(Hdf5Mesh):
    def __init__(self, h5FileName, cellName, kt):
        cell=json.load( open(cellName))
        self.name=cell['name']
        lattice=np.array(cell['lattice'])
        mass=[]
        bc=[] #bound coherent scattering length
        pos=[]
        for id, data in cell['atom'].items():
            for k, v in data.items():
                m, b =getAtomMassBC(k)
                mass.append(m)
                bc.append(b)
                pos.append(np.array(v))
        pos=np.array(pos)
        mass=np.array(mass)
        bc=np.array(bc)

        super().__init__(lattice, mass, pos, bc, kt, h5FileName )
