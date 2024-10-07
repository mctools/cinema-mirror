#!/usr/bin/env python3

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box, Sphere, Tube
from Cinema.Prompt.scorer import VolFluenceHelper, ESpectrumHelper
from Cinema.Prompt.physics import Material
from Cinema.Prompt.gun import IsotropicGun
from Cinema.Prompt.GidiSetting import GidiSetting 
import matplotlib.pyplot as plt
import numpy as np
import time

cdata=GidiSetting()
cdata.setEnableGidi(True)
cdata.setEnableGidiPowerIteration(True)
cdata.setGidiThreshold(5)

class MySim(PromptMPI):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)   

    def makeWorld(self):
        radius = 87.407
        # fuel = Material('freegas::U/18.8gcm3/U_is_U235')
        # fuel = 'freegas::U/18.8gcm3/U_is_0.1000_U238_0.9000_U235;temp=293.6'
        # fuel = Material('freegas::UH999/1.1gcm3/U_is_U235/H_is_H1;temp=293.6') 
        fuel = Material('freegas::U7H993/1.1gcm3/U_is_0.8571428571428572_U238_0.14285714285714285_U235/H_is_H1;temp=293.6') 
        world = Volume("world", Sphere(0, radius+2))
        godiva = Volume('Godiva', Sphere(0, radius), matCfg=fuel)
        world.placeChild('frod', godiva)     

        self.setWorld(world)

sim = MySim(seed=1010)
sim.makeWorld()

gun = IsotropicGun()
gun.setEnergy(3e6)
gun.setPosition([0,0,0])

batchsize = 1e4
# vis or production

# sim.show(gun, 1)

tottime = time.time()
sim.simulate(gun, batchsize)

totCycle = 15
settleCycle = 5
totneutron = 0

for i in range(totCycle):
    t1 = time.time()
    num = sim.simulateSecondStack(batchsize)
    print(f'iteration {i}', f'completed in {time.time()-t1}s')
    if i>=settleCycle:
        totneutron += num
        print('keff', totneutron/(i-settleCycle+1)/batchsize, f'+/- {1./np.sqrt(totneutron)}')


print('keff', totneutron/(totCycle-settleCycle)/batchsize, f'+/- {1./np.sqrt(totneutron)}', f'completed in {time.time()-tottime}s')

