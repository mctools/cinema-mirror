#!/usr/bin/env python3

from Cinema.Prompt import Launcher, Visualiser
from Cinema.Prompt.solid import Box, Tessellated
from Cinema.Prompt.gun import PythonGun

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box, Sphere, Tube, Trapezoid, ArbTrapezoid, Cone
from Cinema.Prompt.gun import PythonGun
from Cinema.Prompt.scorer import WlSpectrum
import numpy as np
import pyvista

from Cinema.Prompt import Prompt
from Cinema.Prompt.geo import Volume

expectWl = [425., 454., 427., 456., 425., 447., 392., 415., 362., 334., 295.,
       276., 271., 234., 243., 194., 192., 183., 157., 156.]


class MySim(Prompt):
    def __init__(self, seed) -> None:
        super().__init__(seed)

    def makeWorld(self):
        world = Volume('world', Box(10, 10, 20))

        dtt = Volume('detector', Box(2, 2, 2))

        t = Tessellated(pyvista.Sphere(radius=1))
        tes = Volume('T', t) 
        dtt.placeChild("Tessellated_TP", tes, Transformation3D(0., 0, 0))

        world.placeChild('detectorPhy', dtt, Transformation3D(0,1,0))

        self.setWorld(world)

sim = MySim(seed=4096)
sim.makeWorld()
gunCfg = "gun=MaxwellianGun;src_w=2;src_h=2;src_z=-10;slit_w=2;slit_h=2;slit_z=1e99;temperature=293;"
sim.show(gunCfg, 100)
# sim.simulate(gunCfg, 1e6)



