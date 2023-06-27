import h5py
import numpy as np
import json

from .calculator import Hdf5Powder
from ..AtomInfo import getAtomMassBC
from ..io.cell import QeXmlCell

class MeshCell(Hdf5Powder):
    def __init__(self, h5FileName, cellName, temperature):
        cell=json.load( open(cellName))
        self.name=cell['name']
        lattice=np.array(cell['lattice'])
        mass=[]
        bc=[] #bound coherent scattering length
        pos=[]
        for id, data in cell['sites'].items():
            for k, v in data.items():
                m, b, _ =getAtomMassBC(k)
                mass.append(m)
                bc.append(b)
                pos.append(np.array(v))
        pos=np.array(pos)
        mass=np.array(mass)
        bc=np.array(bc)
        super().__init__(lattice, mass, pos, bc, temperature, h5FileName )


class MeshQE(Hdf5Powder):
    def __init__(self, h5FileName, qexml, temperature, phonIdx=None):
        qecell = QeXmlCell(qexml)
        self.name=qexml
        lattice=qecell.lattice
        pos = qecell.position
        mass, bc, _ = qecell.getAtomInfo()
        super().__init__(lattice, mass, pos, bc, temperature, h5FileName, phonIdx)
