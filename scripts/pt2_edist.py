#!/usr/bin/env python3

from Cinema.Prompt import Prompt
from Cinema.Prompt.geo import Volume
from Cinema.Prompt.solid import Box, Sphere, Tube
from Cinema.Prompt.scorer import VolFluenceHelper, ESpectrumHelper
from Cinema.Prompt.physics import Material
from Cinema.Prompt.gun import IsotropicGun, SimpleThermalGun, PythonGun
from Cinema.Prompt.centralData import CentralData 
import matplotlib.pyplot as plt
from Cinema.Interface import plotStyle

plotStyle()

cdata=CentralData()
cdata.setGidiThreshold(10)


class MySim(Prompt):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)   

    def makeWorld(self):
        size = 1e-5

        world = Volume("world", Tube(0, size, 1.1e5))
        # lw = Material('freegas::He/1gcm3/He_is_1_He3') 
        lw = Material('freegas::B/1gcm3/B_is_1_B10') 
        # lw = Material('freegas::Li/1gcm3/Li_is_1_Li6') 
        # lw = Material('freegas::C/1gcm3') 
        lw.setBiasAbsp(10)
        lw.setBiasScat(1)
        media = Volume("media", Tube(0, size*0.5, 1e5), matCfg= lw)
        world.placeChild('media', media)

        # VolFluenceHelper('volFlux', max=20e6, numbin=300).make(media)
        ESpectrumHelper('ESpec', min=1e-6, max=20e6, numbin=300, ptstate='EXIT').make(media)
        self.setWorld(world)



sim = MySim(seed=1010)
sim.makeWorld()

# class MyGun(PythonGun):
#     def __init__(self):
#         super().__init__()

#     def samplePosition(self):
#         return 0,0,-1.e5
    
#     def sampleEnergy(self):
#         return 1


# gun = MyGun()

gun = SimpleThermalGun()
gun.setEnergy(1)
gun.setPosition([0,0,-1e5])


# sim.show(gun, 1)


partnum = 1e6
# vis or production
sim.simulate(gun, partnum)

# sim.gatherHistData('volFlux').plot(show=False, log=True)

espec = sim.gatherHistData('ESpec')

# w=espec.getWeight()
# hit=espec.getHit()
# centre = espec.getCentre()

# plt.loglog(centre, w)
# plt.title(str(w.sum()))
# plt.show()

espec.plot(show=True, log=True)


