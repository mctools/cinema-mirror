#! Python3

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


class MySim(PromptMPI):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)
        self.makeWorld()
        
    def makeWorld(self, anyUserParameters=np.zeros(2)):

        world = Volume("world", Box(1000, 1000, 35000))

        self.scorer['PSD1'], vol_PSD1 = self.makePSD('psd_before_guide', -50, 50, -50, 50, 100, 100)
    
        guide1 = makeTrapezoidGuide(1840./ 2, 90./2, 90./2, 90./2, 90./2, 0., 1e-3)
        guide2 = makeTrapezoidGuide(1495./ 2, 90./2, 90./2, 84.196/2, 84.196/2, 2.1, 1e-3)
        guide3 = makeTrapezoidGuide(510./ 2, 84.196/2, 84.196/2, 82.207/2, 82.207/2, 4.1, 1e-3)
        guide4 = makeTrapezoidGuide(795./ 2, 79.867/2, 79.867/2, 76.766/2, 76.766/2, 4.1, 1e-3)
        chopper1 = makeDiskChopper(300, 300 - 74.7, 44.22, 1, 25, 82.94)
        tunnel1 = Volume("tunnel1", Tube(300, 800, 20./2, 0, 360), "B4C.ncmat")
        tunnel2 = Volume("tunnel2", Tube(300, 800, 20./2, 0, 360), "B4C.ncmat")
        # sheild1 = self.makeChopperSheild()
        guide5 = makeTrapezoidGuide(2700./ 2, 75.986/2, 75.986/2, 65.456/2, 65.456/2, 4.1, 1e-3)
        chopper2 = makeDiskChopper(300, 300 - 67., 58.81, 1, 25, 110.79)
        # sheild2 = self.makeChopperSheild()
        guide6 = makeTrapezoidGuide(5800./ 2, 64.676/2, 64.676/2, 42.056/2, 42.056/2, 4.1, 1e-3)
        self.scorer['PSD2'], vol_PSD2 = self.makePSD('psd_before_sample', -100, 100, -100, 100, 100, 100)

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
        # Fixme : ziyi, move this function into det.makeFlatPSD()
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
        vol = Volume('chopper', Box(1000., 1000., 1e-3))
        chp = DiskChopper()
        chp.cfg_rotFreq = 1
        chp.cfg_n = 1
        chp.cfg_phase = 0
        chp.cfg_r = 0
        chp.cfg_theta0 = 0
        vol.setSurface(chp.get_cfg())

        return vol


# class MyGun(PythonGun):
#     def __init__(self):
#         super().__init__()

#     def samplePosition(self):
#         return 0,0,-209


sim = MySim(seed=1010)

# set gun
if True: #could be in either old style or the python style
    gunCfg = "gun=UniModeratorGun;mean_wl=100;range_wl=99;src_w=100;src_h=100;src_z=0;slit_w=100;slit_h=100;slit_z=2250"
    sim.setGun(gunCfg)
else:
    gun = MyGun()
    sim.setGun(gun)

# vis or production
if False:
    sim.show(2000)
else:
    sim.simulate(1e8)

psd1hist = sim.getScorerHist(sim.scorer['PSD1'])
psd2hist = sim.getScorerHist(sim.scorer['PSD2'])

if sim.rank==0:
    # print(f'total {psd1hist.getWeight().sum()} {psd1hist.getHit().sum()}')
    psd1hist.plot(show=True)
    psd2hist.plot(show=True)