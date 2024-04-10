#!/usr/bin/env python3

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box, Sphere, Tube
from Cinema.Prompt.scorer import VolFluenceHelper, ESpectrumHelper
from Cinema.Prompt.physics import Material
from Cinema.Prompt.gun import IsotropicGun
from Cinema.Prompt.centralData import CentralData 
import matplotlib.pyplot as plt

cdata=CentralData()
cdata.setEnableGidi(True)
cdata.setEnableGidiPowerIteration(True)
cdata.setGidiThreshold(4)

class MySim(PromptMPI):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)   

    def makeWorld(self):

        boxhsize = 9000 # 1 inch
        radius = 8740.7
        fuel = Material('freegas::U/0.188gcm3/U_is_U235;temp=293.6') 
        world = Volume("world", Box(boxhsize, boxhsize, boxhsize))
        godiva = Volume('Godiva', Sphere(0, radius), matCfg=fuel)
        world.placeChild('frod', godiva)     

        self.setWorld(world)

sim = MySim(seed=1010)
sim.makeWorld()

gun = IsotropicGun()
gun.setEnergy(1e6)
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


print('keff', totneutron/(totCycle-settleCycle+1)/batchsize)

