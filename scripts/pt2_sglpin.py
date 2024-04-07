#!/usr/bin/env python3

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box, Sphere
from Cinema.Prompt.scorer import VolFluenceHelper, ESpectrumHelper
from Cinema.Prompt.physics import Material
from Cinema.Prompt.gun import IsotropicGun
from Cinema.Prompt.centralData import CentralData 
import matplotlib.pyplot as plt

cdata=CentralData()
cdata.setGidiThreshold(10)
cdata.setEnableGidi(False)


class MySim(PromptMPI):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)   

    def makeWorld(self):

        world = Volume("world", Box(600, 600, 600))
        refboudary = Volume("refboudary", Box(600, 600, 1), surfaceCfg=f'physics=Mirror;m=-1')
        world.placeChild('refboudary', refboudary, transf=Transformation3D(0,0,-400))



        # lw = Material('freegas::He/1gcm3/He_is_1_He3') 
        lw = Material('freegas::B/1gcm3/B_is_1_B10') 
        # lw = Material('freegas::Li/1gcm3/Li_is_1_Li6') 
        # lw = Material('freegas::H2O/1gcm3/H_is_1_H1/O_is_1_O16') 

        lw.setBiasScat(1.)
        lw.setBiasAbsp(1)
        sphere = Volume("sphere", Sphere(0, 300), matCfg=lw, surfaceCfg=f'physics=Mirror;m=-1')
        world.placeChild('sphere', sphere)

        VolFluenceHelper('spct', max=20e6, numbin=300).make(sphere)
        ESpectrumHelper('escap', min=1e-5, max=20e6, ptstate='EXIT').make(sphere)
     

        self.setWorld(world)

sim = MySim(seed=1010)
sim.makeWorld()

gun = IsotropicGun()
gun.setEnergy(1e-5)
gun.setPosition([0,0,-350])

partnum = 1e5
# vis or production

sim.show(gun, 1)

# sim.simulate(gun, partnum)


# destination = 0
# spct = sim.gatherHistData('spct', dst=destination)
# escap= sim.gatherHistData('escap', dst=destination)
# if sim.rank==destination:
#     # escap.plot(log=True)
#     # plt.legend(loc=0)
#     # plt.show()
#     spct.plot(show=True, log=True)
