#!/usr/bin/env python3

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box,Tube
from Cinema.Prompt.scorer import makePSD, ESpectrumHelper, WlSpectrumHelper, TOFHelper, VolFluenceHelper
from Cinema.Prompt.gun import PythonGun
from Cinema.Prompt.Histogram import wl2ekin
from Cinema.Prompt.physics import Material, Mirror
from Cinema.Prompt.component import makeTrapezoidGuide

import numpy as np

# Reproduction of the figure 13 of the paper https://doi.org/10.1016/j.cpc.2023.109004

class MySim(PromptMPI):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)   

    def makeWorld(self):
        world = Volume("world", Box(70, 70, 2100))

        detector1 = Volume("det1", Box(25, 25, 0.0001))
        makePSD('psd1', detector1, 20, 20 )
        world.placeChild("det", detector1, Transformation3D(0., 0., -1000))

        detector2 = Volume("det2", Box(35, 35, 0.0001))
        makePSD('psd2', detector2, 20, 20 )
        world.placeChild("det2", detector2, Transformation3D(0., 0., 1200))

        guide = makeTrapezoidGuide(500, 25,25,25,25, 1.0, 10)
        world.placeChild("guide", guide, Transformation3D(0., 0., -400))

        self.setWorld(world)



sim = MySim(seed=1010)
sim.makeWorld()

# set gun
gunCfg = f'gun=UniModeratorGun;mean_wl=3.39;range_wl=0.3;src_w=50;src_h=50;src_z=-2000;slit_w=50;slit_h=50;slit_z=-1000"'

# vis or production
if False:
    sim.show(1)
else:
    sim.simulate(gunCfg, 1e6)

destination = 0
psd1 = sim.gatherHistData('psd1')
psd2 = sim.gatherHistData('psd2')

if sim.rank==destination:
    psd1.plot(show=False, log=False)
    psd2.plot(show=True, log=False)
