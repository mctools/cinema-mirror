#!/usr/bin/env python3

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box,Tube
from Cinema.Prompt.scorer import  ESpectrumHelper, WlSpectrumHelper, TOFHelper, VolFluenceHelper, PSDHelper
from Cinema.Prompt.gun import PythonGun
from Cinema.Prompt.physics import Material, Mirror
from Cinema.Prompt.gun import UniModeratorGun

import numpy as np


class MySim(PromptMPI):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)   

    def make_symetric_vol(self, ):
        pass

    def makeWorld(self):
        void_cfg = "void.ncmat"
        universe_cfg = 'freegas::H1/1e-26kgm3'

        universe = Material(universe_cfg)
        universe.setBiasScat(1.)
        universe.setBiasAbsp(1.)
        world = Volume("world", Box(800, 800, 800), matCfg=universe)

        simbox = Volume("simbox", Box(600, 600, 600), matCfg=universe)

        name_moderator = 'moderator'
        mat_moderator = Material('LiquidWaterH2O_T293.6K.ncmat;density=1gcm3;temp=293.6')
        sol_moderator = Tube(0, 90, 80)
        vol_moderator = Volume(f"vol_{name_moderator}", sol_moderator, universe)
        # simbox.placeChild(f"pv_{name_moderator}", vol_moderator, Transformation3D().applyRotX(90))

        fe = Material(universe_cfg)
        fe.setBiasScat(1.)
        fe.setBiasAbsp(1.)
        fe_layer = Volume("fe_layer", Tube(100, 300, 80, 0, 180), matCfg=universe)
        simbox.placeChild('pv_fe_layer_b1', fe_layer, Transformation3D().applyRotX(90))
        # simbox.placeChild('pv_fe_layer_b2', fe_layer, Transformation3D().applyRotxyz(90, 180 ,0))
        
        detector2 = Volume("det2", Box(35, 35, 0.0001))
        ESpectrumHelper('espec').make(detector2)
        WlSpectrumHelper('wlspec').make(detector2)
        TOFHelper('tof').make(detector2)

        obxz = PSDHelper('PSDXZ',-400,400,200,-400,400,200,ptstate='ENTRY',psdtype='XZ',isGlobal=True)
        obxz.make(vol_moderator)
        obxz.make(fe_layer)
        obxz.make(simbox)

        world.placeChild('pv_simbox', simbox)
        self.setWorld(world)


class PositionTestGun(PythonGun):
    def samplePosition(self):
        x = np.random.uniform(-310,310)
        return np.array([0, 0, -610])

    def sampleDirection(self):
        x = np.random.uniform(-1,1)
        return np.array([x, 0, 1])
gun = PositionTestGun()


sim = MySim(seed=1010)
sim.makeWorld()


# vis or production
if True:
    sim.show(gun, 100)
else:
    sim.simulate(gun, 1e4)
    draw_xz = sim.gatherHistData('PSDXZ')
    destination = 0
    if sim.rank==destination:
        draw_xz.plot(1)