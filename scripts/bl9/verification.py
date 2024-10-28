#!/usr/bin/env python3

# Verify building prompt guide section by McstasGuideData
# by checking the entry and exit opening size


import scipy.stats
from Cinema.Prompt import Launcher, Visualiser
from Cinema.Prompt.solid import Box, Tessellated
from Cinema.Prompt.gun import PythonGun

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box, Sphere, Tube, Trapezoid, ArbTrapezoid, Cone
from Cinema.Prompt.gun import PythonGun, MCPLGun
from Cinema.Prompt.scorer import WlSpectrum, PSDHelper
import numpy as np

from Cinema.Prompt import Prompt
from Cinema.Prompt.geo import Volume

import json
import pyvista as pv
import os
import scipy

from mcgd import McstasGuideData, GuideSection, GuideSectionCollection

entryx = 120
entryy = 250
exitx = 10
exity = 80
length = 1000

class MySim(PromptMPI):
    def __init__(self, seed) -> None:
        super().__init__(seed)

    def makeWorld(self):
        world = Volume('world', Box(2*entryx, 2*entryy, length))
        gbox = Volume('gbox', Box(1.5*entryx, 1.5*entryy, 0.8*length))
        sec1_data = McstasGuideData(entryx, entryy, exitx, exity, length, -0.5*length)

        # check if thickness impacts
        thickness = 1e-4
        section1 = GuideSection.from_McstasGuideData(sec1_data, thickness)

        psdxy = PSDHelper('PSDXY',-1.2*entryx,1.2*entryx,200,-1.2*entryy,1.2*entryy,200,isGlobal=True)
        psdxz = PSDHelper('PSDXZ',-1.2*entryx,1.2*entryx,200,-0.6*length,0.6*length,200,psdtype='XZ',isGlobal=True)

        for wall in section1.pvc:
            psdxy.make(wall.logicalVolume)
            psdxz.make(wall.logicalVolume)

        section1.placeInto(gbox)
        # section2.placeInto(world)
        world.placeChild('gboxpv', gbox)
        self.setWorld(world)

if __name__ == "__main__":
    class PositionTestGun(PythonGun):
        def samplePosition(self):
            x = np.random.uniform(-entryx, entryx)
            y = np.random.uniform(-entryy, entryy)
            return np.array([x, y, -0.75*length])
    gun = PositionTestGun()
    sim = MySim(seed=102)
    sim.makeWorld()
    # sim.show(gun, 100, zscale=1)
    sim.simulate(gun, 1e6)
    xy = sim.gatherHistData('PSDXY')
    xz = sim.gatherHistData('PSDXZ')
    if sim.rank == 0:
        def check(scorer, section_type):
            plt = scorer.plot(show=False,log=False)
            plt.figure()
            plt.title(f'{section_type} Height profile')
            plt.xticks(np.linspace(-entryy,entryy,10))
            plt.grid(True)
            plt.plot(scorer.getCentre()[0], scorer.getWeight().sum(0),)
            plt.figure()
            plt.title(f'{section_type} Width profile')
            plt.xticks(np.linspace(-entryx,entryx,10))
            plt.grid(True)
            plt.plot(scorer.getCentre()[1], scorer.getWeight().sum(1))
            plt.show()

        check(xy, 'XY')
        check(xz, 'XZ')
    

