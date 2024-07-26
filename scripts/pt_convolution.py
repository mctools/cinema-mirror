#!/usr/bin/env python3

from Cinema.Prompt import Prompt
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Sphere, Box
from Cinema.Prompt.scorer import ESpectrumHelper, MultiScatCounter
from Cinema.Prompt.physics import Material
from Cinema.Prompt.gun import PythonGun
from Cinema.Prompt.centralData import CentralData 
import matplotlib.pyplot as plt
from Cinema.Interface import plotStyle
import numpy as np

cdata=CentralData()
cdata.setEnableGidi(False)
cdata.setGammaTransport(False)

class MyGun(PythonGun):
    def __init__(self, pdg, ekin):
        super().__init__(pdg)
        self.ekin = ekin

    def samplePosition(self):
        return 0,0,-100.
    
    def sampleEnergy(self):
        if isinstance(self.ekin, list):
            r=np.random.uniform(self.ekin[0],  self.ekin[1], 1)[0]
            return r
        else:
            return self.ekin

class MySim(Prompt):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)   

    def makeWorld(self):

        world = Volume("world", Box(200, 200, 400))

        lw = "physics=idealElaScat;xs_barn=1;density_per_aa3=5;energy_transfer_eV=0.02"
        # lw = Material('solid::CH2/1gcm3')

        monitor = Volume("mon", Box(1,1,1.e-6))
        eRange = [0., 0.2]
        binNum = (eRange[1] - eRange[0]) * 1000   #0.001 per bin
        ESpectrumHelper('incident_energy', eRange[0], eRange[1], int(binNum), groupID=0, linear=True).make(monitor)
        world.placeChild('monPV', monitor, Transformation3D(0.,0.,-50))

        sample = Volume("sample", Box(1,1,1), matCfg=lw)
        ms = MultiScatCounter()
        ms.make(sample)
        world.placeChild('samplePV', sample)
        

        detMat = Material("freegas::He/0.0001347gcm3/He_is_He3") # Employ this (and also ABSORB pts) to boarden energy spectrum by detector efficiency
        detector = Volume("DttLV", Sphere(50, 51, 0 , 2*np.pi, starttheta=np.deg2rad(1), deltatheta=np.pi-np.deg2rad(2)))
        dttSc = ESpectrumHelper('energy_gp0', eRange[0], eRange[1], int(binNum), ptstate='ENTRY', energyTransfer=True, groupID=0, linear=True)
        dttSc.make(detector)
        dttSc.addScatterCounter(ms, 1)
        world.placeChild('DttPV', detector)

        self.setWorld(world)



sim = MySim(seed=1010)
sim.makeWorld()

gun = MyGun(2112, [0.11, 0.12])

if False:
    partnum = 1000
    sim.show(gun, partnum)
else:
    partnum = 1e6
    sim.simulate(gun, partnum)

dtt0 = sim.gatherHistData('energy_gp0')
mon0 = sim.gatherHistData('incident_energy')
print('TOTAL: ', dtt0.getHit().sum())

# mon0.plot(show=True)
dtt0.plot(show=True, log=False)
# plt.plot(dtt0.getEdge()[:-1], dtt0.getWeight(), label='@detector')
# plt.plot(mon0.getEdge()[:-1], mon0.getWeight(), label='@monitor')
# plt.legend()
# plotStyle()
# plt.show()

