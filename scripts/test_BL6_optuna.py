#!/usr/bin/env python3

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.solid import Box, Tube, Trapezoid
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.component import makeTrapezoidGuide, makeDiskChopper
from Cinema.Prompt.scorer import PSD

from Cinema.Prompt.gun import PythonGun
import numpy as np

# ---- inputs from mcstas guide_BL6.instr
# double	ll1=	1.84;
# double	ll2=	1.495;
# double	ll3=	0.51;
# double	ll4=	0.795;
# double	ll5=	2.7;
# double	ll6=	5.8;
		
# double	ww1=	0.09;
# double	ww2=	0.09;
# double	ww3=	0.084196;
# double	ww4=	0.079867;
# double	ww5=	0.075986;
# double	ww6=	0.064676;
		
# double	hh1=	0.09;
# double	hh2=	0.084196;
# double	hh3=	0.082207;
# double	hh4=	0.076766;
# double	hh5=	0.065456;
# double	hh6=	0.042056;


# double  T0length=0.48;   /* T0length=0.1 */
# double  T0Phase=64.67;  
# double  T0Frequency=50;
# double dtheta = 45;
# double mos=120;    
# int ord_mcm=1;   
# ----


class MySim(Prompt):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)
        
    def makeWorld(self, g2_in, g2_out, g3_in, g3_out, g4_in, g4_out, g5_in, g5_out, g6_in, g6_out):

        world = Volume("world", Box(1000, 1000, 35000))

        self.scorer['PSD1'], vol_PSD1 = self.makePSD('psd_before_guide', -50, 50, -50, 50, 100, 100)

        g1_in = 90.*0.5
        g1_out = 90.*0.5

        # g2_in = g1_out
        # g2_out = 84.196*0.5

        # g3_in = g2_out
        # g3_out = 82.207*0.5

        # g4_in = 79.867*0.5
        # g4_out = 76.766*0.5

        # g5_in = 75.986*0.5
        # g5_out = 65.456*0.5

        # g6_in = 64.676*0.5
        # g6_out = 42.056*0.5

        guide1 = makeTrapezoidGuide(1840./ 2, g1_in, g1_in, g1_out, g1_out, 0.)
        guide2 = makeTrapezoidGuide(1495./ 2, g2_in, g2_in, g2_out, g2_out, 2.1)
        guide3 = makeTrapezoidGuide(510./ 2,  g3_in, g3_in, g3_out, g3_out, 4.1)
        guide4 = makeTrapezoidGuide(795./ 2,  g4_in, g4_in, g4_out, g4_out, 4.1)
        chopper1 = makeDiskChopper(300, 300 - 74.7, 44.22, 1, 25, 82.94)
        tunnel1 = Volume("tunnel1", Tube(300, 800, 20./2, 0, 360), "B4C.ncmat")
        tunnel2 = Volume("tunnel2", Tube(300, 800, 20./2, 0, 360), "B4C.ncmat")
        # sheild1 = self.makeChopperSheild()
        guide5 = makeTrapezoidGuide(2700./ 2, g5_in, g5_in, g5_out, g5_out, 4.1)
        chopper2 = makeDiskChopper(300, 300 - 67., 58.81, 1, 25, 110.79)
        # sheild2 = self.makeChopperSheild()
        guide6 = makeTrapezoidGuide(5800./ 2, g6_in, g6_in, g6_out, g6_out, 4.1)
        self.scorer['PSD2'], vol_PSD2 = self.makePSD('psd_before_sample', -50, 50, -50, 50, 100, 100)

        world.placeChild("monitor_bf", vol_PSD1, Transformation3D(0., 0., 2300.))
        world.placeChild("guide1", guide1, Transformation3D(0., 0., 1840. / 2 + 2304.))
        world.placeChild("guide2", guide2, Transformation3D(0., 0., 1495. / 2 + 2304.))
        world.placeChild("guide3", guide3, Transformation3D(0., 0., 510. / 2 + 2304.))
        world.placeChild("guide4", guide4, Transformation3D(0., 0., 795. / 2 + 2304.))
        world.placeChild("chopper1", chopper1, Transformation3D(0., - (300 - 74.7 / 2), 6800.))
        world.placeChild("tunnel1_", tunnel1, Transformation3D(0., - (300 - 74.7 / 2), 6800.))
        # world.placeChild("chopper1sheild", sheild1, Transformation3D(0., 0., 6800. + 1e-2 / 2))
        world.placeChild("guide5", guide5, Transformation3D(0., 0., 2700. / 2 + 6800.))
        world.placeChild("chopper2", chopper2, Transformation3D(0., - (300 - 67. / 2), 9070.))
        world.placeChild("tunnel2_", tunnel2, Transformation3D(0., - (300 - 67. / 2), 9070.))
        # world.placeChild("chopper2sheild", sheild2, Transformation3D(0., 0., 9070. + 1e-2 / 2))
        world.placeChild("guide6", guide6, Transformation3D(0., 0., 5800. / 2 + 9120.))
        world.placeChild("monitor_sample", vol_PSD2, Transformation3D(0., 0., 17000.))
        self.l.setWorld(world)

    def makePSD(self, name, xmin, xmax, ymin, ymax, numbin_x, numbin_y):

        det = PSD()
        det.cfg_name = name
        det.cfg_xmin = xmin
        det.cfg_xmax = xmax
        det.cfg_numbin_x = numbin_x

        det.cfg_ymin = ymin
        det.cfg_ymax = ymax
        det.cfg_numbin_y = numbin_y
        cfg = det.makeCfg()
        vol = Volume(name, Box(1000,1000,1e-3))
        vol.addScorer(cfg)
        return cfg, vol
 
    def makeChopperSheild(self):

        from Cinema.Prompt.component import DiskChopper
        vol = Volume('chopper', Box(1000., 1000.,1e-3))
        chp = DiskChopper()
        chp.cfg_rotFreq = 1
        chp.cfg_n = 1
        chp.cfg_phase = 0
        chp.cfg_r = 0
        chp.cfg_theta0 = 0
        vol.setSurface(chp.get_cfg())

        return vol


# sim = MySim(seed=1010)
# sim.makeWorld(30.84343978203833, 16.672921795397997,
#                  20.82267768215388, 40.97299173939973, 
#                  40.23749625310302, 39.766095648519695,
#                  41.34080420248210, 38.571013724431396,
#                  40.64210145268589, 35.14687175862491,
#                  40.89239043649286, 24.26095639821142)
# # set gun
# if True: #could be in either old style or the python style
#     gunCfg = "gun=UniModeratorGun;mean_wl=100;range_wl=99;src_w=100;src_h=100;src_z=0;slit_w=100;slit_h=100;slit_z=2250"
#     sim.setGun(gunCfg)
# else:
#     gun = MyGun()
#     sim.setGun(gun)

# # vis or production
# if False:
#     sim.show(2000)
# else:
#     sim.simulate(1e6)
                 

# psd1hist = sim.getScorerHist(sim.scorer['PSD1'])
# psd2hist = sim.getScorerHist(sim.scorer['PSD2'])

# # print(f'total {psd1hist.getWeight().sum()} {psd1hist.getHit().sum()}')
# psd1hist.plot(show=False)
# psd2hist.plot(show=True)



from botorch.settings import validate_input_scaling
import optuna
import numpy as np


sim = MySim()
incident = 1e5

from optuna import trial as trial_module
def objective(trial : trial_module.Trial) -> float:
    sim.clear() 
    
    g2_in  = trial.suggest_float("g2_in", 15, 50)
    g2_out = trial.suggest_float("g2_out", 15, 50)
    g3_in  = trial.suggest_float("g3_in", 15, 50)
    g3_out = trial.suggest_float("g3_out", 15, 50)
    g4_in  = trial.suggest_float("g4_in", 15, 50)
    g4_out = trial.suggest_float("g4_out", 15, 50)
    g5_in  = trial.suggest_float("g5_in", 15, 50)
    g5_out = trial.suggest_float("g5_out", 15, 50)
    g6_in  = trial.suggest_float("g6_in", 15, 50)
    g6_out = trial.suggest_float("g6_out", 15, 50)

    sim.makeWorld(g2_in, g2_out, g3_in, g3_out, g4_in, g4_out, g5_in, g5_out, g6_in, g6_out)
    # set gun
    gunCfg = "gun=UniModeratorGun;mean_wl=100;range_wl=99;src_w=100;src_h=100;src_z=0;slit_w=100;slit_h=100;slit_z=2250"
    sim.setGun(gunCfg)
    sim.simulate(incident)
    # sim.show(100)
    hist2 = sim.getScorerHist(sim.scorer['PSD2'])
    return -hist2.getWeight().sum() # it is a minimisation optimiser


if __name__ == "__main__":
    # Show warnings from BoTorch such as unnormalized input data warnings.
    validate_input_scaling(True)

    sampler = optuna.integration.BoTorchSampler(n_startup_trials=500,)

    study = optuna.create_study(
        directions=["minimize"],
        sampler=sampler,)
    
    study.optimize(objective, n_trials=2000, timeout=6000)

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