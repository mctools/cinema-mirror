import h5py
import numpy as np
import json

from .Hdf5Mesh import Hdf5Mesh
from ..AtomInfo import getAtomMassBC
from ..io.cell import XmlCell

class MeshCell(Hdf5Mesh):
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


class MeshQE(Hdf5Mesh):
    def __init__(self, h5FileName, qexml, temperature):
        qecell = XmlCell(qexml)
        self.name=qexml
        lattice=qecell.lattice
        pos = qecell.position
        mass=[]
        bc=[] #bound coherent scattering length
        for ele in self.element:
            m, b, _ =getAtomMassBC(ele)
            mass.append(m)
            bc.append(b)
        mass=np.array(mass)
        bc=np.array(bc)
        super().__init__(lattice, mass, pos, bc, temperature, h5FileName )
