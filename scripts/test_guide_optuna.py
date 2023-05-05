#!/usr/bin/env python3
# coding: utf-8

from Cinema.Prompt import Prompt, PromptMPI, Optimiser
from Cinema.Prompt.solid import Box, Tube, Trapezoid
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.component import makeTrapezoidGuide
from Cinema.Prompt.scorer import PSD
import optuna
import numpy as np

class MySim(Prompt):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)

    def makeWorld(self, par):
        print('making world with paramters', par)

        # define scorers
        det = PSD()
        det.cfg_name = 'apsd'
        det.cfg_xmin = -25.
        det.cfg_xmax = 25
        det.cfg_numbin_x = 20

        det.cfg_ymin = -25.
        det.cfg_ymax = 25
        det.cfg_numbin_y = 20
        self.scorer['PSD1'] = det.makeCfg()

        det.cfg_xmin=-10
        det.cfg_xmax=10
        det.cfg_ymin=-10
        det.cfg_ymax=10 
        self.scorer['PSD2'] = det.makeCfg()
      

        # Geometry
        air = "freegas::N78O22/1.225kgm3"
        worldsize = 6500.
        world = Volume("world", Box(worldsize*0.5, worldsize*0.5, worldsize*0.5))
        # world.setMaterial("freegas::N78O22/1.225e3kgm3")

        det1 = Volume("detector", Box(50, 50, 0.01) )
        det1.addScorer(self.scorer['PSD1'])

        world.placeChild('det1', det1, Transformation3D(0, 0, 1000))
        world.placeChild('guide', makeTrapezoidGuide(500., par.get('x'), par.get('x'), par.get('y'), par.get('y'), 3.), Transformation3D(0, 0, 1600))
     
        det2 = Volume("detector", Box(10, 10, 0.01))
        det2.addScorer(self.scorer['PSD2'] )
        world.placeChild('det2', det2, Transformation3D(0, 0, 2200))

        self.setWorld(world)


class GuideOpt(Optimiser):
    def __init__(self, sim, optunaNum=100000) -> None:
        super().__init__(sim, optunaNum)
        self.addParameter('x', lower = 5, upper = 50, promptval = 10)
        self.addParameter('y', 5, 50, 10)

    def objective(self, trial):
        self.sim.clear() 
        p=self.getParameters(trial)
        self.sim.makeWorld(p)
        self.sim.simulate(self.trailNeutronNum)
        hist2 = self.sim.getScorerHist(self.sim.scorer['PSD2'])
        return hist2.getWeight().sum() 



sim = MySim()
gunCfg = "gun=UniModeratorGun;src_w=50;src_h=50;src_z=0;slit_w=50;slit_h=50;slit_z=1100;mean_wl=10.39"
sim.setGun(gunCfg)

opt = GuideOpt(sim)
opt.trailNeutronNum=1e4
# opt.optimize_botorch(n_trials = 100)
opt.optimize(n_trials = 100)

opt.analysis()

# or we can see the initial geometry
# opt.visInitialGeometry()

