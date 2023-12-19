#!/usr/bin/env python3

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box, Tube, Polyhedron
from Cinema.Prompt.scorer import makePSD, ESpectrumHelper, WlSpectrumHelper, TOFHelper, VolFluenceHelper
from Cinema.Prompt.gun import PythonGun
from Cinema.Prompt.histogram import wl2ekin
from Cinema.Prompt.physics import Material, Mirror
from Cinema.Prompt.component import makeTrapezoidGuide
from Cinema.Prompt.gun import UniModeratorGun, IsotropicGun

import numpy as np

# Reproduction of the figure 13 of the paper https://doi.org/10.1016/j.cpc.2023.109004

class MySim(Prompt):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)   

    def makeWorld(self):

        world = Volume("world",  Box(100,100,100), matCfg='Cd.ncmat')
        
        pd = Volume('child', Polyhedron([-50, 0., 60], [0,0,0.], [20., 20, 50] ))
        
        inner = Volume('pd_child', Box(10,10,10))
        pd.placeChild('pd_child', inner)
        
        world.placeChild('det', 
                         pd, 
                         Transformation3D(0., 0., 0))

        self.setWorld(pd)



sim = MySim(seed=1010)
sim.makeWorld()


gun = IsotropicGun()


# vis or production
if True:
    sim.show(gun, 10)
else:
    sim.simulate(gun, 1e6)

# destination = 0
# psd1 = sim.gatherHistData('psd1', dst=destination)
# psd2 = sim.gatherHistData('psd2', dst=destination)
# wlspec = sim.gatherHistData('wlspec', dst=destination)
# if sim.rank==destination:
#     psd1.plot(show=True, log=False)
#     psd2.plot(show=True, log=False)
#     wlspec.plot(show=True, log=False)
