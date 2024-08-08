#!/usr/bin/env python3

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box,Tube, Sphere
from Cinema.Prompt.scorer import DirectSqwHelper
from Cinema.Prompt.histogram import wl2ekin
from Cinema.Prompt.physics import Material, Mirror
from Cinema.Prompt.gun import IsotropicGun, PythonGun

import numpy as np


class MySim(PromptMPI):
    def __init__(self, mod_smp_dist, mean_ekin, seed=4096) -> None:
        super().__init__(seed)   
        self.mod_smp_dist = mod_smp_dist
        self.mean_ekin = mean_ekin

    def makeWorld(self):
        world = Volume("world", Box(2000, 2000, 21000))
        # mat = 'physics=idealElaScat;xs_barn=1;density_per_aa3=5;energy_transfer_eV=0.03'
        mat = 'skeleton_c1_vol_77K.ncmat;bragg=0'
        # mat = 'bzscope_c1_vol_77K.ncmat;bragg=0'
        sample = Volume("sample", Sphere(0, 10), matCfg=mat)
        world.placeChild('samplePV', sample)

        deg = np.pi/180.

        monitor = Volume("mon", Sphere(1499.99, 1500, starttheta=1.*deg, deltatheta=178*deg ))
        helper = DirectSqwHelper('sqw', self.mod_smp_dist, self.mean_ekin, [0,0,1.], [0,0,0],
                 qmin = 1e-1, qmax = 20, num_qbin = 150, 
                 ekinmin=-0.1, ekinmax=0.1,  num_ebin = 100,
                 pdg = 2112, groupID  = 0, ptstate = 'ENTRY')
        helper.make(monitor)

        world.placeChild('monPV', monitor, Transformation3D(0, 0., 0))

       
        

        self.setWorld(world)


class MyGun(PythonGun):
    def __init__(self, mod_smp_dist, ekin ):
        super().__init__(2112)
        self.ekin = ekin
        self.mod_smp_dist = mod_smp_dist

    def samplePosition(self):
        return 0,0,-self.mod_smp_dist
    
    def sampleEnergy(self):
        if isinstance(self.ekin, list):
            r=np.random.uniform(self.ekin[0], self.ekin[1], 1)[0]
            return r
        else:
            return self.ekin

mod_smp_dist = 20000
mean_ekin = 0.05

gun = MyGun(mod_smp_dist, mean_ekin)

sim = MySim(mod_smp_dist, mean_ekin, seed=1010)
sim.makeWorld()

# sim.show(gun, 100)

sim.simulate(gun, 1e7)
res = sim.gatherHistData('sqw')
res.plot(False, log=False)
res.plot(True, log=True)
