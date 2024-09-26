#!/usr/bin/env python3

from Cinema.Prompt import Prompt
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box, Tube
from Cinema.Prompt.scorer import ESpectrumHelper, MultiScatCounter
from Cinema.Prompt.physics import Material
from Cinema.Prompt.gun import SimpleThermalGun
from Cinema.Prompt.GidiSetting import GidiSetting 
import numpy as np

cdata=GidiSetting()
cdata.setEnableGidi(False)
cdata.setGammaTransport(False)

g_scatNum = 10

class MySim(Prompt):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)   

    def makeEspScatNum(self, vol, scatterCounter, interNum):
        esp0 = ESpectrumHelper(f'Scatter{interNum}', 1e-6, 20e6, 100)
        esp0.make(vol)
        esp0.addScatterCounter(scatterCounter, interNum)
        
    def makeWorld(self):

        world = Volume("world", Box(400, 400, 400))

        lw = Material('freegas::H2O/1gcm3/H_is_H1/O_is_O16') 

        lw.setBiasScat(1.)
        lw.setBiasAbsp(1.)
        sample = Volume("sample", Box(1,1,1), matCfg=lw)
        ms_counter = MultiScatCounter()
        ms_counter.make(sample)
        world.placeChild('sphere', sample)

        detector = Volume("DttLV", Box(100,100,1))
        for i in range(g_scatNum):
            self.makeEspScatNum(detector, ms_counter, i)


        world.placeChild('Dtt1', detector, Transformation3D(0,-200*1/np.sqrt(2),200*1/np.sqrt(2)).applyRotX(45), scorerGroup=0)
        world.placeChild('Dtt2', detector, Transformation3D(0,200*1/np.sqrt(2),200*1/np.sqrt(2)).applyRotX(-45), scorerGroup=0)
        self.setWorld(world)



sim = MySim(seed=1010)
sim.makeWorld()

gun = SimpleThermalGun()
gun.setEnergy(1)
gun.setPosition([0,0,-150])


if 0:
    partnum = 100
    sim.show(gun, partnum)
else:
    partnum = 1e6
    sim.simulate(gun, partnum)

scores = []
for j in range(g_scatNum):
    dtt0 = sim.gatherHistData(f'Scatter{j}')
    score = dtt0.getHit().sum()
    print(f'Scatter {j}: ', score)
    scores.append(score)
sc = np.array(scores)[1:]
np.testing.assert_allclose(sc.sum(), 78840.0)
np.testing.assert_allclose(scores[0], 0)

# from unittest import TestCase
# t = TestCase()
# t.assertTrue(scores[0]==sc.sum())
# spct.plot(show=True, log=True)

