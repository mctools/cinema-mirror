#!/usr/bin/env python3

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box,Tube
from Cinema.Prompt.scorer import ESpectrumHelper,MultiScatCounter, WlSpectrumHelper, TOFHelper, VolFluenceHelper, PSDHelper, DirectSqwHelper,DirectSqHelper, KillMCPLHelper
from Cinema.Prompt.gun import PythonGun, SimpleThermalGun, MaxwellianGun
from Cinema.Prompt.histogram import wl2ekin
from Cinema.Prompt.physics import Material, Mirror

import matplotlib.pyplot as plt
import numpy as np

print(" ")

def pprint_dict(dict, title="", unit='mm'):
    print(title)
    for key, value in dict.items():
        print(f"{key:<30} → {str(value):>10}{unit:>3}")
    print(" ")

def move_dist_zero_at_sample(dict):
    dict_out = {}
    pos_sample = dict["sample_changer"]
    for key, value in dict.items():
        dict_out[f"dist2sample_{key}"] = value - pos_sample
    return dict_out

# >> parameters
## >> distance to moderator (mm)
dict_distance = {
"slit_A1"          :       7500 ,
"monitor2"         :       10884,
"slit_A2"          :       11231,
"slit_A3_short"    :       12675,
"slit_A3_long" 	   :       12931,
"sample_changer"   :       13000,
"monitor3"         :       13485,
"window"           :       13635,
"main_detector"    :       15200, # 18200
}
dict_distance["beam_stop"] = dict_distance["main_detector"]  - 1000 # todo: beam stop to detector distance

pprint_dict(dict_distance, "Origin paras: ")

dist2sample = move_dist_zero_at_sample(dict_distance)
pprint_dict(dist2sample,"Distance to sample: ")

## >> detector size (mm)
detector_tube_diameter = 8.5
detector_tube_number = 120
detector_tube_length = 1000
detector_bins_axes_length = 200 
detector_psd_horiz = detector_tube_diameter * 120 # todo: check space between two tube
detector_psd_verti = detector_tube_length

## >> beam stop
### max among config phi4 phi6 phi8 A3=4,6,8 (mm)
beam_stop_diameter = 8 # todo: check
beam_stop_thickness = 10  # todo: beamstop thickness

# mod_sam_dist = 12000
gun_pos = -dict_distance.get("sample_changer")
sam_pos = np.array([0,0,0])
# det_radius_mm = 400.
# beamstop_radius_mm = .001
# det_pos = 6000.

class MySim(PromptMPI):
    def __init__(self, seed=4096, sample_thickness=1, wl=1) -> None:
        super().__init__(seed)
        self.sample = self.make_sample(sample_thickness)
        self.ms = MultiScatCounter()
        self.ms.make(self.sample)
        self.makeWorld(wl)

    def make_sample(self, sample_thickness):
        # matCfg_sample = Material('LiquidWaterH2O_T293.6K.ncmat')
        # matCfg_sample = Material('Al2O3_sg167_Corundum.ncmat')
        # matCfg_sample = Material('PTWaterH2O_T293.6K.ncmat')

        # matCfg_sample = Material('PTHeavyWater_T293.6K.ncmat;ucnmode=refine:0.01eV')
        # matCfg_sample = Material('PTHeavyWater_T293.6K.ncmat')

        # matCfg_sample = Material('LiquidHeavyWaterD2O_T293.6K.ncmat')V_sg229.ncmat
        matCfg_sample = Material('nanodiamond.ncmat')
        matCfg_sample.setBiasScat(2.0)
        sample = Volume('sample', Box(10, 10, sample_thickness), matCfg = matCfg_sample)
        return sample
    
    def make_beamstop(self):
        bs_vol = Tube(0, beam_stop_diameter * 0.5, beam_stop_thickness) 
        bs = Volume("beamstop", bs_vol, "solid::B4C/2.52gcm3/B_is_0.95_B10_0.05_B11")
        return bs

    def make_simple_main_detector(self,wl):
        hx = detector_psd_horiz * 0.5
        hy = detector_psd_verti * 0.5

        msdist = dict_distance.get("sample_changer")
        detector = Volume("det", Box(hx, hy, 1))
        helper = PSDHelper('psd', -hx, hx, 100,  -hy, hy, 100)
        helper.make(detector)
        helper.addScatterCounter(self.ms, 1)
        helper2 = PSDHelper('psd2', -hx, hx, 100,  -hy, hy, 100)
        helper2.make(detector)

        ESpectrumHelper('espec').make(detector)
        WlSpectrumHelper('wlspec').make(detector)
        TOFHelper('tof', max=50e-3).make(detector)
        DirectSqwHelper('sqw', msdist, wl2ekin(wl), sample_position=sam_pos, qmin=1e-1, 
                        qmax=5, num_qbin=100, ekinmin=-0.1, ekinmax=0.1, num_ebin=110 ).make(detector)
        helpersqw = DirectSqwHelper('sqw_s', msdist, wl2ekin(wl), sample_position=sam_pos, qmin=1e-3, 
                        qmax=.1, num_qbin=20, ekinmin=-.01, ekinmax=.01, num_ebin=101, logx=True )
        helpersqw.linear=False
        helpersqw.make(detector)

        helperSq = DirectSqHelper('sq', qmin=2e-3, qmax=0.1,
                                  numbin=50, distanceMS=msdist, 
                                  linear=False)
        helperSq.make(detector)
        return detector

    def makeWorld(self,wl):
        
        world = Volume("world", Box(600, 600, 14000))
       
        world.placeChild("sample", self.sample, Transformation3D(0., 0., dist2sample.get("dist2sample_sample_changer")))
        world.placeChild("det", self.make_simple_main_detector(wl), Transformation3D(0., 0., dist2sample.get("dist2sample_main_detector")))
        world.placeChild("phy_beamstop", self.make_beamstop(), Transformation3D(0., 0., dist2sample.get("dist2sample_beam_stop")))
        # self.kill = KillMCPLHelper('part_gen', 2112)
        # self.kill.make(detector)
        self.setWorld(world)

def sans_run(wl, n, t, det_pos, divergence, pyGun=True):
    sim = MySim(seed=1010)
    class MyGun(PythonGun):
        def __init__(self, pdg):
            super().__init__(pdg)

        def sampleEnergy(self):
            return wl2ekin(wl)

        def samplePosition(self):
            return np.array(gun_pos)
        
        def sampleDirection(self):
            x = np.random.random() * divergence / mod_sam_dist
            y = np.random.random() * divergence / mod_sam_dist
            return np.array([x, y, 1])

    sim.makeWorld(wl,t,detpos=det_pos)

    usePythonGun=pyGun

    if usePythonGun:
        gun = MyGun(2112)
    else:
        gun = SimpleThermalGun()
        gun.setPosition(gun_pos) 
        gun.setDirection([0, 0, 1])
        gun.setWavelength(wl)

# vis or production
    if False:
        sim.show(gun, 100)
    else:
        sim.simulate(gun, n)
    sqw = sim.gatherHistData('sqw')
    sqw_s = sim.gatherHistData('sqw_s')
    sq = sim.gatherHistData('sq')

    psd = sim.gatherHistData('psd') # 1 scatter
    psd2 = sim.gatherHistData('psd2') # all
    espec = sim.gatherHistData('espec')
    wlspec = sim.gatherHistData('wlspec')
    tof = sim.gatherHistData('tof')
    all = psd2.getAccWeight()
    signal = psd.getAccWeight()
    res = signal
    return sim, sqw_s, sq


def main():
    wl = 1
    sim = MySim(seed=1010, sample_thickness=1, wl=wl)
    class MyGun(PythonGun):
        def __init__(self, pdg, meanwl):
            super().__init__(pdg)
            self.meanwl = meanwl

        def sampleEnergy(self):
            wl = np.random.random() + 0.5 * self.meanwl
            return wl2ekin(wl)

        def samplePosition(self):
            x = (np.random.random() - 0.5) * 10 # todo: check source
            y = (np.random.random() - 0.5) * 10
            return np.array([x, y, gun_pos])
        
        def sampleDirection(self):
            return np.array([0, 0, 1])
        
    numNeutron = 1e8
    usePythonGun = True

    if usePythonGun:
        gun = MyGun(2112, wl)
    else:
        gun = SimpleThermalGun()
        gun.setPosition(gun_pos) 
        gun.setDirection([0, 0, 1])
        gun.setWavelength(wl)
    if True:
        sim.show(gun, 100, byMat=1, addLegend=1)
    else:
        sim.simulate(gun, numNeutron)
    sqw = sim.gatherHistData('sqw')
    sqw_s = sim.gatherHistData('sqw_s')

    psd = sim.gatherHistData('psd') # 1 scatter
    psd2 = sim.gatherHistData('psd2') # all
    espec = sim.gatherHistData('espec')
    wlspec = sim.gatherHistData('wlspec')
    tof = sim.gatherHistData('tof')
    all = psd2.getAccWeight()
    signal = psd.getAccWeight()
    if sim.rank==0: 
    #     psd.save('psd.h5')
    #     espec.save('espec.h5')
    #     wlspec.save('wlspec.h5')
    #     tof.save('tof.h5')
    #     sqw.save('sqw.h5')
    #     sqw_s.save('sqw_s.h5')
        sq = sqw.getWeight().sum(1)
        q = sqw.getCentre()[0]
        fig = plt.figure(0)
        plt.plot(q,sq)
        fig.savefig('sq.pdf')
        sqw.savefig('sqw.pdf', log=True)

        psd.plot(show=False)
        sqw.plot(show=False, dynrange=1e-10)
        sqw_s.plot(show=False, dynrange=1e-10, logx=True)
        plt.figure()
        espec.plot(show=False, log=True)
        plt.figure()
        wlspec.plot(show=False,  log=[False, True])
        plt.figure()
        tof.plot(show=False)

if __name__ == "__main__":
    main()