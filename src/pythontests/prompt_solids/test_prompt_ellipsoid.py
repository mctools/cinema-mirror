#!/usr/bin/env python3

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box,Tube, Ellipsoid
from Cinema.Prompt.scorer import makePSD, ESpectrumHelper, WlSpectrumHelper, TOFHelper
from Cinema.Prompt.gun import PythonGun
from Cinema.Prompt.histogram import wl2ekin
from Cinema.Prompt.physics import Material, Mirror
from Cinema.Prompt.component import makeTrapezoidGuide
from Cinema.Prompt.gun import UniModeratorGun

import numpy as np

class MySim(PromptMPI):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)   

    def makeWorld(self):
        universe = Material('freegas::H1/1e-26kgm3')
        universe.setBiasScat(1.)
        universe.setBiasAbsp(1.)

        world = Volume("world", Box(200, 200, 2100), matCfg=universe)

        detector1 = Volume("det1", Box(100, 100, 0.0001))
        makePSD('psd1', detector1, 50, 50 )
        world.placeChild("det", detector1, Transformation3D(0., 0., -1000))

        detector2 = Volume("det2", Box(100, 100, 0.0001))
        makePSD('psd2', detector2, 50, 50 )
        world.placeChild("det2", detector2, Transformation3D(0., 0., 1000))

        ESpectrumHelper('espec').make(detector2)
        WlSpectrumHelper('wlspec').make(detector2)
        TOFHelper('tof').make(detector2)

        # guide = makeTrapezoidGuide(500, 25,25,25,25, 1.0, 10)
        wallMat = 'solid::Cd/8.65gcm3'
        guideWall = Volume('guideWall', Box(100, 100., 1000, ), matCfg=wallMat)
        mirror = Mirror(10)
        guide = Volume('guideEllipsoid', Ellipsoid(70, 70, 1800, -1000, 1000), surfaceCfg=mirror.cfg)
        guideWall.placeChild("guide", guide)
        world.placeChild("guideWall", guideWall)

        self.setWorld(world)



sim = MySim(seed=1010)
sim.makeWorld()


gun = UniModeratorGun()
gun.setWlMean(3.39)
gun.setWlRange(0.3)
gun.setSlit([80,80,-1500])
gun.setSource([80,80,-2000])


# vis or production
if True:
    sim.show(gun, 10)
else:
    sim.simulate(gun, 1e7)

destination = 0
psd1 = sim.gatherHistData('psd1', dst=destination)
psd2 = sim.gatherHistData('psd2', dst=destination)
wlspec = sim.gatherHistData('wlspec', dst=destination)
if sim.rank==destination:
    psd1.plot(show=True, log=False, title='GUIDE-')
    psd2.plot(show=True, log=False, title='GUIDE+')
    # wlspec.plot(show=True, log=False)
