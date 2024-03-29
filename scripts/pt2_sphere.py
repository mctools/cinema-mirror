#!/usr/bin/env python3

from Cinema.Prompt import Prompt
from Cinema.Prompt.geo import Volume
from Cinema.Prompt.solid import Box, Sphere
from Cinema.Prompt.scorer import VolFluenceHelper, ESpectrumHelper
from Cinema.Prompt.physics import Material
from Cinema.Prompt.gun import IsotropicGun
from Cinema.Prompt.centralData import CentralData 

cdata=CentralData()
cdata.setGidiThreshold(1e-10)


class MySim(Prompt):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)   

    def makeWorld(self):

        world = Volume("world", Box(400, 400, 400))

        # lw = Material('freegas::He/1gcm3/He_is_1_He3') 
        # lw = Material('freegas::B/1gcm3/B_is_1_B10') 
        # lw = Material('freegas::Li/1gcm3/Li_is_1_Li6') 
        lw = Material('freegas::H/1gcm3') 

        lw.setBiasScat(1.)
        lw.setBiasAbsp(1.)
        sphere = Volume("sphere", Sphere(0, 300), matCfg=lw)
        world.placeChild('sphere', sphere)

        VolFluenceHelper('spct', max=20e6, numbin=300).make(sphere)
        # ESpectrumHelper('spct', min=1, max=20e6, numbin=80, ptstate='EXIT').make(sphere)

        self.setWorld(world)

sim = MySim(seed=1010)
sim.makeWorld()

gun = IsotropicGun()
gun.setEnergy(1e6)

partnum = 1e5
# vis or production
sim.simulate(gun, partnum)

spct = sim.gatherHistData('spct')
# print(spct.getWeight().sum())
spct.plot(show=True, log=True)


