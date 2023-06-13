#!/usr/bin/env python3

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.solid import Box, Tube, Trapezoid
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.scorer import makeFlatPSD
from Cinema.Prompt.component import EntityArray, makeTrapezoidGuide, makeDiskChopper, make2CurveAnalyser


from Cinema.Prompt.gun import PythonGun
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as scipyRot
# ---- inputs from mcstas guide_BL6.instr
# ll1 = 1.84;
# ll2 = 1.495;
# ll3 = 0.51;
# ll4 = 0.795;
# ll5 = 2.7;
# ll6 = 5.8;

ww1 = 0.09 * 1000 * 0.5
ww2 = 0.09 * 1000 * 0.5
ww3 = 0.084196 * 1000 * 0.5
ww4 = 0.079867 * 1000 * 0.5
ww5 = 0.075986 * 1000 * 0.5
ww6 = 0.064676 * 1000 * 0.5 

hh1 = 0.09 * 1000 * 0.5
hh2 = 0.084196 * 1000 * 0.5
hh3 = 0.082207 * 1000 * 0.5
hh4 = 0.076766 * 1000 * 0.5
hh5 = 0.065456 * 1000 * 0.5
hh6 = 0.042056 * 1000 * 0.5 


# double  T0length=0.48;   /* T0length=0.1 */
# double  T0Phase=64.67;  
# double  T0Frequency=50;
# double dtheta = 45;
# double mos=120;    
# int ord_mcm=1;   
# ----

# --- inputs from pptx ---
dist_z_g1 = 2360.
dist_z_g2 = 4200.
dist_z_g3 = 5695.
dist_z_g4 = 6805.
dist_z_g5 = 7800.
dist_z_g6 = 10500.   # deviation

m1 = 0.
m2 = 2.1
m3 = 4.1
m4 = 4.1
m5 = 4.1
m6 = 4.1
# --- end ---
class MySim(Prompt):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)
        
    def makeWorld(self, g2_in, g2_out, g3_in, g3_out, g4_in, g4_out, g5_in, g5_out, g6_in, g6_out):

        world = Volume("world", Box(3000, 3000, 35000))

        self.scorer['PSD1'], vol_PSD1 = makeFlatPSD('psd_before_guide', 200, 200, 100, 100)

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

        guide1 = makeTrapezoidGuide(1840.* 0.5, g1_in, g1_in, g1_out, g1_out, m1)
        guide2 = makeTrapezoidGuide(1495.* 0.5, g2_in, g2_in, g2_out, g2_out, m2)
        guide3 = makeTrapezoidGuide(510.* 0.5,  g3_in, g3_in, g3_out, g3_out, m3)
        guide4 = makeTrapezoidGuide(795.* 0.5,  g4_in, g4_in, g4_out, g4_out, m4)
        chopper1 = makeDiskChopper(300, 300 - 74.7, 44.22, 1, 25, 82.94)
        tunnel1 = Volume("tunnel1", Tube(300, 800, 50.* 0.5, 0, 360), "B4C.ncmat") # fixme: overlap
        tunnel2 = Volume("tunnel2", Tube(300, 800, 50.* 0.5, 0, 360), "B4C.ncmat") # fixme: overlap
        # sheild1 = self.makeChopperSheild()
        guide5 = makeTrapezoidGuide(2700.* 0.5, g5_in, g5_in, g5_out, g5_out, m5)
        chopper2 = makeDiskChopper(300, 300 - 67., 58.81, 1, 25, 110.79)
        # sheild2 = self.makeChopperSheild()
        sample_mat = "physics=ncrystal;nccfg='Ge_sg227.ncmat;mos=0.267deg;incoh_elas=0;inelas=0;dir1=@crys_hkl:5,1,1@lab:0,0,1;dir2=@crys_hkl:0,1,-1@lab:1,0,0';scatter_bias=5.0;abs_bias=5.0"
        guide6 = makeTrapezoidGuide(5800.* 0.5, g6_in, g6_in, g6_out, g6_out, m6)
        sample = Volume("sample_Al", Box(40, 40, 40), sample_mat)
        self.scorer['PSD2'], vol_PSD2 = makeFlatPSD('psd_before_sample', 200, 200, 100, 100)
        cry_mat = "physics=ncrystal;nccfg='C_sg194_pyrolytic_graphite.ncmat;mos=2.5deg;dir1=@crys_hkl:0,0,2@lab:0,-1,0;dir2=@crys_hkl:1,0,0@lab:1,0,0;lcaxis=0,0,1';scatter_bias=5.0;abs_bias=5.0"
        crystal_plate = Volume("cry", Box(6, 6, 1), 'solid::Cd/8.65gcm3', surfaceCfg='physics=Mirror;m=2')

        world.placeChild("monitor_bf", vol_PSD1, self.translationZ(2300.))
        world.placeChild("guide1", guide1, self.translationZ(1840. * 0.5 + dist_z_g1))
        world.placeChild("guide2", guide2, self.translationZ(1495. * 0.5 + dist_z_g2))
        world.placeChild("guide3", guide3, self.translationZ(510. * 0.5 + dist_z_g3))
        world.placeChild("guide4", guide4, self.translationZ(795. * 0.5 + dist_z_g4))
        # world.placeChild("chopper1", chopper1, Transformation3D(0., - (300. - 74.7 * 0.5), 6800.))
        # world.placeChild("tunnel1_", tunnel1, Transformation3D(0., - (300 - 74.7 * 0.5), 6800.))
        # world.placeChild("chopper1sheild", sheild1, Transformation3D(0., 0., 6800. + 1e-2 * 0.5))
        world.placeChild("guide5", guide5, self.translationZ(2700. * 0.5 + dist_z_g5))
        # world.placeChild("chopper2", chopper2, Transformation3D(0., - (300 - 67. * 0.5), 9070.)) # input != mcstas
        # world.placeChild("tunnel2_", tunnel2, Transformation3D(0., - (300 - 67. * 0.5), 9070.)) # input != mcstas
        # world.placeChild("chopper2sheild", sheild2, Transformation3D(0., 0., 9070. + 1e-2 * 0.5))
        world.placeChild("guide6", guide6, self.translationZ(5800. * 0.5 + dist_z_g6))
        world.placeChild("sample", sample, Transformation3D(0., 0., 18000.).applyRotY(0))
        analyser = EntityArray(crystal_plate, [20,20], spacings = [13, 13], 
                               refFrame=Transformation3D(0., 0., 0.).applyRotxyz(0., 180., 0.))
        analyser.make_trape_plane(80, 280, 350, 500, 500)
        # analyser.make_plane(500., 500.)
        # print(f'Anchors are : {analyser.volAncs}')
        
        arrayAna = EntityArray(analyser, refFrame=Transformation3D(0., -300., 17200.).applyRotZ(180))
        arrayAna.make()
        arrayAna.make_rotate_z(0, 0, 45, 6)
        # arrayAna.rotate_z(0, 0, 45)
        # arrayAna.rotate_z(0, 0, -45, 3)
        arrayAna.reflect('XY', 18000)
        # arrayAna.repeat([0,1,0], 300, 2)
        world.placeArray(arrayAna)

        # arrayAna.repeat([1,0,0], 300, 2)
        # arrayAna_vert = EntityArray(analyser, refFrame=Transformation3D(180., 400., 17200.))
        # arrayAna_vert.make()
        # arrayAna_vert.repeat([0,1,0], 300, 2)
        # arrayAna_vert.reflect()
        # world.placeArray(arrayAna_vert)

        world.placeChild("monitor_sample", vol_PSD2, Transformation3D(1500., 0., 19000., 90., 90., 0.))
        self.l.setWorld(world)

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

    def translationZ(self, dist_z):
        
        return Transformation3D(0., 0., dist_z)
    
    def translationChopper(self, dist_y, dist_z):

        """

        centering

        """
        pass

    def quadSurface(self, coordinates, a, b, c, d):
        """
        Following: ax^2 + by^2 + cz^2 + d = 0
        Generally: ax^2 + by^2 + cz^2 + 2fyz + 2gzx + 2hxy + 2px + 2qy + 2rz + d = 0

        Args:
            a (_type_): _description_
            b (_type_): _description_
            c (_type_): _description_
            d (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = coordinates[0]
        y = coordinates[1]
        z = coordinates[2]
        return a * x ** 2 + b * y ** 2 + c * z ** 2 + d

# class Analyser(Array):s

#     def __init__(self, num_x, num_y, plate_size_x, plate_size_y, gap_x, gap_y, algorithm) -> None:
#         """_summary_

#         Args:
#             size (list): _description_
#             algorithm (str): "RTA". RTA denotes ray tracing approximation.
#         """
#         array_size = [num_x, num_y]
#         spacings = np.array([gap_x + plate_size_x, gap_y + plate_size_y])
#         super().__init__(array_size, spacings)
#         self.algorithm = algorithm
#         self.set_algorithm()

#     def set_algorithm(self):

#         if self.algorithm == 'RTA':
#             self.rayTrace()
#         else:
#             raise ValueError("Allowable values is 'RTA'.")

#     def rayTrace(self):
#         self.shape = 'PLANE'
        

if True:
    sim = MySim(seed=1010)
    sim.makeWorld(
        # ww1, hh1,
        ww2, hh2,
        ww3, hh3,
        ww4, hh4,
        ww5, hh5,
        ww6, hh6
        # 48.20394391889674, 40.57624712255829,
        # 42.50467953195299, 36.25224824022611,
        # 35.748035108437605, 29.487747978839167,
        # 34.81729138952778, 41.59840248022938,
        # 47.39264459608555, 38.51865250117852,

    )
    # set gun
    if True: #could be in either old style or the python style
        gunCfg = "gun=UniModeratorGun;mean_wl=200;range_wl=1;src_w=100;src_h=100;src_z=0;slit_w=100;slit_h=100;slit_z=2250"
        # gunCfg = "gun=SimpleThermalGun;energy=0.000002045105;position=0,0,-1500;direction=0,0,1"
        sim.setGun(gunCfg)
    else:
        gun = MyGun()
        sim.setGun(gun)

    # vis or production
    if True:
        sim.show(100)
    else:
        sim.simulate(1e6)
                    

    psd1hist = sim.getScorerHist('PSD1')
    psd2hist = sim.getScorerHist('PSD2')

    # print(f'total {psd1hist.getWeight().sum()} {psd1hist.getHit().sum()}')
    psd1hist.plot(show=True)
    psd2hist.plot(show=True)
    print(f'PSD1 weight: {psd1hist.getWeight().sum()}')
    print(f'PSD2 weight: {psd2hist.getWeight().sum()}')
    print(f'Efficiency: {psd2hist.getWeight().sum()/psd1hist.getWeight().sum()*100}%')
else:
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
        
        study.optimize(objective, n_trials=100, timeout=6000)

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