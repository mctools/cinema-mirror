
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

    def estPhonMesh(self, size=200.):
        res = (size/(2*np.pi/self.abc)/len(self.num)).astype(int)
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
        return (self.lattice, self.reduced_pos, self.num)
    
    def getAtomSymbols(self):
        raise NotImplementedError()


    def getAtomInfo(self):
        symb = self.getAtomSymbols()

        mass=[]
        bc=[] #bound coherent scattering length
        num=[]

        for ele in symb:
            m, b, n =getAtomMassBC(ele)
            mass.append(m)
            bc.append(b)
            num.append(n)

        return np.array(mass), np.array(bc), np.array(num)

    

class QeXmlCell(CellBase):
    def __init__(self, filename, au2Aa=0.529177248994098):
        super().__init__()
        self.reduced_pos=[]
        self.num=[]
        self.element=[]
        self.lattice=np.zeros((3,3))

        def internal(subtree):
            if list(subtree):
                for child in list(subtree):
                    if child.tag=='atom':
                        ele = child.attrib['name']
                        atominfo=getAtomMassBC(ele)
                        self.element.append(ele)
                        self.num.append(atominfo[2])
                        self.reduced_pos.append(np.fromstring(child.text, sep=' '))
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
        self.reduced_pos = np.array(self.reduced_pos)*au2Aa # it is in the abs unit
        for i in range(self.reduced_pos.shape[0]):
            self.reduced_pos[i] = invlatt.dot(self.reduced_pos[i])

        self.totmagn = float( (root.findall('./output/magnetization/total')[0]).text )
        self.lattice_reci = np.linalg.inv(self.lattice.T)*2*np.pi

        # print(self.lattice_reci, self.element)

    def qreduced2abs(self, r):
        return r.dot(self.lattice_reci)
       
    def qabs2reduced(self, q):
        fac = 1./(2*np.pi)
        return q.dot(self.lattice.T)*fac
    
    def getAtomSymbols(self):
        return self.element




class MPCell(CellBase):
    def __init__(self, filename):
        super().__init__()
        if isinstance(filename, dict):
            mat_dict = filename
        else:
            with open(filename, 'r') as fp:
                mat_dict = json.load(fp)

        self.reduced_pos=[]
        self.num=[]
        self.element=[]

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
            self.reduced_pos.append(site['abc'])
            elename = site['species'][0]['element']
            atominfo=getAtomMassBC(elename)
            self.element.append(elename)
            self.num.append(atominfo[2])

    def getTotalMagnetic(self):
        return self.totMagn

    def getSpacegourpNum(self):
        return self.spacegroupnum
