
import numpy as np
import json
import xml.etree.ElementTree as ET
from ..AtomInfo import getAtomMassBC

class CellBase():
    def __init__(self):
        self.lattice = np.zeros([3, 3])
        self.abc = None
        self.position=[]
        self.num=[]

    def estSupercellDim(self, size=10.):
        return (size//self.abc).astype(int)

    def estRelaxKpoint(self, size=20.):
        return (size//self.abc).astype(int)

    def estSupercellKpoint(self, size=30., supercellDim=None):
        if supercellDim is None:
            return (size//(self.abc*self.estSupercellDim())).astype(int)
        else:
            return (size//(self.abc*supercellDim)).astype(int)

    def getCell(self):
        return (self.lattice, self.position, self.num)


# atom {'name': 'Fe', 'index': '1'} 0.000000000000000e0 0.000000000000000e0 0.000000000000000e0
# cell {}
#
# a1 {} -2.313781319104671e0 2.313781319104671e0 3.249128552770355e0
# a2 {} 2.313781319104671e0 -2.313781319104671e0 3.249128552770354e0
# a3 {} 2.313781319104671e0 2.313781319104671e0 -3.249128552770354e0


class XmlCell(CellBase):
    def __init__(self, filename, au2Aa=0.529177248994098):
        super().__init__()
        def internal(subtree):
            if list(subtree):
                for child in list(subtree):
                    if child.tag=='atom':
                        ele = child.attrib['name']
                        atominfo=getAtomMassBC(ele)
                        self.num.append(atominfo[2])
                        self.position.append(np.fromstring(child.text, sep=' '))
                    elif child.tag=='a1':
                        self.lattice[0] = np.fromstring(child.text, sep=' ')*au2Aa
                    elif child.tag=='a2':
                        self.lattice[1] = np.fromstring(child.text, sep=' ')*au2Aa
                    elif child.tag=='a3':
                        self.lattice[2] = np.fromstring(child.text, sep=' ')*au2Aa
                    internal(child)

        root = ET.parse(filename)
        info = root.findall('./output/atomic_structure')
        if len(info)!=1:
            raise RuntimeError('./output/atomic_structure is not unique')
        internal(info[0])
        self.abc = np.linalg.norm(self.lattice, axis=1)



class JsonCell(CellBase):
    def __init__(self, filename):
        super().__init__()
        with open(filename, 'r') as fp:
            mat_dict = json.load(fp)
            structure = mat_dict['structure']
            self.lattice = (structure['lattice']['matrix'])
            self.abc = np.linalg.norm(self.lattice, axis=1)
            # self.sites={}
            # sites
            for i, site in enumerate(structure['sites']):
                if len(site['species'])!=1:
                    raise RuntimeError('occu is not unity')
                # self.sites[i]={site['species'][0]['element']: site['abc'] }
                self.position.append(site['abc'])
                atominfo=getAtomMassBC(site['species'][0]['element'])
                self.num.append(atominfo[2])
