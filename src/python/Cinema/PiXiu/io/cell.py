
import numpy as np
import json
import xml.etree.ElementTree as ET
from ..AtomInfo import getAtomMassBC

class CellBase():
    def __init__(self, lattice=None):
        if lattice is not None:
            if lattice.shape != (3,3):
                raise RuntimeError('wrong lattice shape')
            self.lattice = lattice
            self.abc = np.linalg.norm(self.lattice, axis=1)
        else:
            self.lattice = np.zeros([3, 3])
            self.abc = None
        self.position=[]
        self.num=[]
        self.element=[]

    def estSupercellDim(self, size=10.):
        res = (size//self.abc).astype(int)
        res[np.where(res==0)]=1
        return res

    def estRelaxKpoint(self, size=20., forceOdd=False):
        res = (size//self.abc).astype(int)
        res[np.where(res==0)]=1
        if forceOdd:
            res[np.where(res%2==0)]+=1
        return res

    def estMesh(self, size=200.):
        res = (size//self.abc).astype(int)
        res[np.where(res==0)]=1
        return res

    def estSupercellKpoint(self, size=30., supercellDim=None):
        if supercellDim is None:
            res = (size//(self.abc*self.estSupercellDim())).astype(int)
        else:
            res = (size//(self.abc*supercellDim)).astype(int)
        res[np.where(res==0)]=1
        return res

    def getCell(self):
        return (self.lattice, self.position, self.num)


class XmlCell(CellBase):
    def __init__(self, filename, au2Aa=0.529177248994098):
        super().__init__()
        def internal(subtree):
            if list(subtree):
                for child in list(subtree):
                    if child.tag=='atom':
                        ele = child.attrib['name']
                        atominfo=getAtomMassBC(ele)
                        self.element.append(ele)
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
        invlatt = np.linalg.inv(self.lattice).T
        self.position = np.array(self.position)*au2Aa
        for i in range(self.position.shape[0]):
            self.position[i] = invlatt.dot(self.position[i])

        self.totmagn = float( (root.findall('./output/magnetization/total')[0]).text )
        print(self.position, self.element)





class JsonCell(CellBase):
    def __init__(self, filename):
        super().__init__()
        if isinstance(filename, dict):
            mat_dict = filename
        else:
            with open(filename, 'r') as fp:
                mat_dict = json.load(fp)

        self.totMagn = mat_dict.get('total_magnetization')
        self.spacegroupnum =  mat_dict['spacegroup']['number']
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
            elename = site['species'][0]['element']
            atominfo=getAtomMassBC(elename)
            self.element.append(elename)
            self.num.append(atominfo[2])

    def getTotalMagnetic(self):
        return self.totMagn

    def getSpacegourpNum(self):
        return self.spacegroupnum
