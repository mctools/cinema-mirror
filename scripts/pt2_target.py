#!/usr/bin/env python3

from Cinema.Prompt import PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box, Sphere
from Cinema.Prompt.scorer import *
from Cinema.Prompt.physics import Material
from Cinema.Prompt.gun import IsotropicGun, SimpleThermalGun
from Cinema.Prompt.centralData import CentralData 
import matplotlib.pyplot as plt

cdata=CentralData()
cdata.setGidiThreshold(10)
cdata.setEnableGidi(True)


class MySim(PromptMPI):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)   

    def makeWorld(self, bias):
        world = Volume("world", Box(100, 100, 100))
        # lw = Material('freegas::He/1gcm3/He_is_1_He3') 
        lw = Material('freegas::Th/1gcm3') 
        # lw = Material('freegas::Li/1gcm3/Li_is_1_Li6') 
        # lw = Material('freegas::V/1gcm3') 
        lw.setBiasAbsp(bias)
        lw.setBiasScat(bias) # for NCrystal < 10eV
        media = Volume("media", Box(100, 100, 10), matCfg= lw)
        world.placeChild('media', media)

        box  = Volume('box', Box(99,10,1e-6))

        world.placeChild('box', box, Transformation3D(0., 15., 99))
        ESpectrumHelper('ESpec', min=1e-5, max=20e6, numbin=100).make(box)
        TOFHelper('TOF', min=0, max=1e-8, numbin=100).make(box)

        self.setWorld(world)



sim = MySim(seed=1010)


gun = SimpleThermalGun()
gun.setEnergy(10e5)
gun.setPosition([0,0,-99])

partnum = 1e7
# vis or production
# sim.show(gun, 100)

destination = 0

for i in range(3):  
    sim.clear()
    sim.makeWorld(i*2+1)
    sim.simulate(gun, partnum)
    # tof=sim.gatherHistData('TOF', dst=destination)
    espec=sim.gatherHistData('ESpec', dst=destination)

    if sim.rank==destination:
        # tof.plot(show=False, log=False)
        espec.plot(show=False, log=True)

if sim.rank==destination:
    plt.xlabel('energy, eV')
    plt.ylabel('count')
    plt.legend(loc=0)
    plt.show()