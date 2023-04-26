#!/usr/bin/env python3
# coding: utf-8

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.solid import Box, Tube, Trapezoid
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.component import makeTrapezoidGuide
from Cinema.Prompt.scorer import PSD
from botorch.settings import validate_input_scaling
import optuna
import numpy as np



class MySim(Prompt):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)

    def makeWorld(self, x, y):
        print('making world', x, y)

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
        world.placeChild('guide', makeTrapezoidGuide(500., x, x, y, y, 3.), Transformation3D(0, 0, 1600))
     
        det2 = Volume("detector", Box(10, 10, 0.01))
        det2.addScorer(self.scorer['PSD2'] )
        world.placeChild('det2', det2, Transformation3D(0, 0, 2200))

        self.setWorld(world)

sim = MySim()
incident = 1e6

def objective(trial) -> float:
    sim.clear() 
    
    x = trial.suggest_float("x", 5, 50)
    y = trial.suggest_float("y",  5, 50)
   
    sim.makeWorld(x, y)
    # set gun
    gunCfg = "gun=UniModeratorGun;src_w=50;src_h=50;src_z=0;slit_w=50;slit_h=50;slit_z=1100;mean_wl=10.39"
    sim.setGun(gunCfg)
    sim.simulate(incident)
    # sim.show(100)
    hist2 = sim.getScorerHist(sim.scorer['PSD2'])
    return -hist2.getWeight().sum() # it is a minimisation optimiser


if __name__ == "__main__":
    # Show warnings from BoTorch such as unnormalized input data warnings.
    validate_input_scaling(True)

    sampler = optuna.integration.BoTorchSampler(
        n_startup_trials=10,
    )
    study = optuna.create_study(
        directions=["minimize"],
        sampler=sampler,
    )
    study.optimize(objective, n_trials=32, timeout=600)

    print("Number of finished trials: ", len(study.trials))

    print("Pareto front:")

    trials = sorted(study.best_trials, key=lambda t: t.values)

    for trial in trials:
        print("  Trial#{}".format(trial.number))
        print(
            "    Values: Values={}".format(
                trial.values
            )
        )
        print("    Params: {}".format(trial.params))