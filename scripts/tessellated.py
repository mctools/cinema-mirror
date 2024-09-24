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

from Cinema.Prompt import Prompt
from Cinema.Prompt.geo import Volume

expectWl = [425., 454., 427., 456., 425., 447., 392., 415., 362., 334., 295.,
       276., 271., 234., 243., 194., 192., 183., 157., 156.]


class MySim(Prompt):
    def __init__(self, seed) -> None:
        super().__init__(seed)

    def makeWorld(self):
        world = Volume('world', Box(50, 50, 200))
        import pyvista
        sphere = pyvista.Sphere(radius=10)
        t = Tessellated(sphere.faces, sphere.points)

        tes = Volume('T', t) 
        world.placeChild("Tessellated_TP", tes, Transformation3D(0., 0., 0))


        self.setWorld(world)



sim = MySim(seed=4096)
sim.makeWorld()
gunCfg = "gun=MaxwellianGun;src_w=20;src_h=20;src_z=-100;slit_w=20;slit_h=20;slit_z=1e99;temperature=293;"
# gunCfg = "gun=UniModeratorGun;mean_wl=1;range_wl=0.001;src_w=2;src_h=2;src_z=-100;slit_w=2;slit_h=2;slit_z=1e99"
sim.show(gunCfg, 100)



