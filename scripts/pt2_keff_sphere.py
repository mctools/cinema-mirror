#!/usr/bin/env python3

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box, Sphere, Tube
from Cinema.Prompt.scorer import VolFluenceHelper, ESpectrumHelper
from Cinema.Prompt.physics import Material
from Cinema.Prompt.gun import IsotropicGun
from Cinema.Prompt.GidiSetting import GidiSetting 
import matplotlib.pyplot as plt

cdata=GidiSetting()
cdata.setEnableGidi(True)
cdata.setEnableGidiPowerIteration(True)
cdata.setGidiThreshold(5)

class MySim(PromptMPI):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)   

    def makeWorld(self):
        radius = 87.407
        # fuel = Material('freegas::U/18gcm3/U_is_U235')
        fuel = Material('freegas::UH999/1.1gcm3/U_is_U235/H_is_H1;temp=293.6') 
        world = Volume("world", Sphere(0, radius+2))
        godiva = Volume('Godiva', Sphere(0, radius), matCfg=fuel)
        world.placeChild('frod', godiva)     

        self.setWorld(world)

sim = MySim(seed=1010)
sim.makeWorld()

gun = IsotropicGun()
gun.setEnergy(10e6)
gun.setPosition([0,0,0])

batchsize = 1e4
# vis or production

# sim.show(gun, 1)

sim.simulate(gun, batchsize)

totCycle = 250
settleCycle = 50
totneutron = 0

for i in range(totCycle):
    num = sim.simulateSecondStack(batchsize)
    print(f'iteration {i}')
    if i>=settleCycle:
        totneutron += num
        print('keff', totneutron/(i-settleCycle+1)/batchsize)


print('keff', totneutron/(totCycle-settleCycle)/batchsize)

