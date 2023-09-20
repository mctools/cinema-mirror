#!/usr/bin/env python3

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box, Sphere
from Cinema.Prompt.gun import PythonGun

import numpy as np


class MySim(PromptMPI):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)

    def makeWorld(self, anyUserParameters=np.zeros(2)):
        # matCfg_sample = "physics=ncrystal;nccfg='LiquidHeavyWaterD2O_T293.6K.ncmat';scatter_bias=1;abs_bias=1"
        matCfg_sample = "physics=ncrystal;nccfg='LiquidHeavyWaterD2O_nsk0_T293.6K.ncmat';scatter_bias=1;abs_bias=1"

        world = Volume("world", Box(1600, 1600, 20000))
        sample = Volume('sample', Sphere(0,1), matCfg = matCfg_sample)

        deg = np.pi/180.
        detector = Volume("Det", Sphere(1500-.001, 1500, starttheta=5*deg, deltatheta=170.*deg))
      
        world.placeChild("sample", sample, Transformation3D(0., 0., 8000), 1)
        world.placeChild("detector", detector, Transformation3D(0., 0., 8000), 1)

        # detector
        self.scorerCfg='Scorer=DeltaMomentum;name=PofQ_HW;sample_pos=0,0,8000;beam_dir=0,0,1;dist=27000;ptstate=EXIT;method=0;scatnum=-1;min=0.1;max=30.1;numbin=50;linear=yes'

        # self.scorerCfg = scorerCfg_detwl
        detector.addScorer(self.scorerCfg)

        # detector = Volume("Det", Box(400, 400, 0.0001), matCfg = matCfg_sample)
        # detector.addScorer(scorerCfg_detpsd)100
        # detector.addScorer(scorerCfg_detwl)

        # sample = Volume("sample", Box(20,20,20))
        # water = Volume('sbox', Box(2,2,2), matCfg = matCfg_sample)

        # for i in range(10):
        #     sample.placeChild("ssample", water, Transformation3D(0,0,2.2*(i-5.0)), scorerGroup=i)

        # world.placeChild("physicalbox", detector, Transformation3D(0., 0., 50), 1)
        # world.placeChild("physicalbox2", sample, Transformation3D(-150, 45, 45))
        self.setWorld(world)


class MyGun(PythonGun):
    def __init__(self):
        super().__init__()

    def samplePosition(self):
        return 0,0,-209


sim = MySim(seed=1010)
sim.makeWorld()

# set gun
if True: #could be in either old style or the python style
    gunCfg = "gun=SimpleThermalGun;position=0,0,-19000;direction=0,0,1"
    sim.setGun(gunCfg)
else:
    gun = MyGun()
    sim.setGun(gun)

# vis or production
if False:
    sim.show(1000)
else:
    sim.simulate(1e8)

hist = sim.getScorerHist(sim.scorerCfg, raw=True)

if sim.rank == 0:    
    hist.savefig('test.pdf')
    hist.plot(show=True)


