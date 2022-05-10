
import numpy as np
import json


class Cell:
    def __init__(self, filename):
        with open(filename, 'r') as fp:
            mat_dict = json.load(fp)
            structure = mat_dict['structure']
            self.lattice = np.array(structure['lattice']['matrix'])
            self.abc = np.linalg.norm(self.lattice, axis=1)
            self.sites={}
            # sites
            for i, site in enumerate(structure['sites']):
                if len(site['species'])!=1:
                    raise RuntimeError('occu is not unity')
                self.sites[i]={site['species'][0]['element']: site['abc'] }

    def estSupercellDim(self, size=10.):
        return size//self.abc

    def estRelaxKpoint(self, size=20.):
        return size//self.abc

    def estSupercellKpoint(self, size=30., supercellDim=None):
        if supercellDim is None:
            return size//(self.abc*self.estSupercellDim())
        else:
            return size//(self.abc*supercellDim)
