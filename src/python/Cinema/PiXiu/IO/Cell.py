
import numpy as np
import json

from ..AtomInfo import getAtomMassBC

class Cell:
    def __init__(self, filename):
        with open(filename, 'r') as fp:
            mat_dict = json.load(fp)
            structure = mat_dict['structure']
            self.lattice = (structure['lattice']['matrix'])
            self.abc = np.linalg.norm(self.lattice, axis=1)
            self.sites={}
            self.position=[]
            self.num=[]
            # sites
            for i, site in enumerate(structure['sites']):
                if len(site['species'])!=1:
                    raise RuntimeError('occu is not unity')
                self.sites[i]={site['species'][0]['element']: site['abc'] }
                self.position.append(site['abc'])
                info=getAtomMassBC(site['species'][0]['element'])
                self.num.append(info[2])

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
