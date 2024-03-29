#!/usr/bin/env python3

from Cinema.Prompt import Prompt
from Cinema.Prompt.geo import Volume
from Cinema.Prompt.solid import Box, Sphere
from Cinema.Prompt.scorer import VolFluenceHelper, ESpectrumHelper
from Cinema.Prompt.physics import Material
from Cinema.Prompt.gun import IsotropicGun, SimpleThermalGun
from Cinema.Prompt.centralData import CentralData 

cdata=CentralData()
cdata.setGidiThreshold(10)


class MySim(Prompt):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)   

    def makeWorld(self):

        world = Volume("world", Box(1, 1, 1.1e5))

        # lw = Material('freegas::He/1gcm3/He_is_1_He3') 
        # lw = Material('freegas::B/1gcm3/B_is_1_B10') 
        # lw = Material('freegas::Li/1gcm3/Li_is_1_Li6') 
        lw = Material('freegas::H2O/1gcm3') 
        lw.setBiasAbsp(10)
        lw.setBiasScat(1)
        media = Volume("media", Box(1, 1, 1.1e5), matCfg= lw)
        world.placeChild('media', media)

        VolFluenceHelper('volFlux', max=20e6, numbin=300).make(media)
        ESpectrumHelper('ESpec', min=1, max=10, numbin=300, ptstate='EXIT').make(media)
        self.setWorld(world)



sim = MySim(seed=1010)
sim.makeWorld()


gun = SimpleThermalGun()
gun.setEnergy(1e6)
gun.setPosition([0,0,-1e5])

partnum = 1e5
# vis or production
sim.simulate(gun, partnum)

# sim.gatherHistData('volFlux').plot(show=False, log=True)

sim.gatherHistData('ESpec').plot(show=True, log=False)


