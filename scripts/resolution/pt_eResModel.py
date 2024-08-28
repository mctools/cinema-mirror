#!/usr/bin/env python3

# Model description from
# https://www.sciencedirect.com/science/article/pii/S016890021301423X


from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box,Tube, Sphere
from Cinema.Prompt.scorer import DirectSqwHelper, MultiScatCounter, TOFHelper, ESpectrumHelper
from Cinema.Prompt.histogram import wl2ekin
from Cinema.Prompt.physics import Material, Mirror
from Cinema.Prompt.gun import IsotropicGun, PythonGun
from Cinema.Prompt.component import makeDiskChopper

import numpy as np


class MySim(PromptMPI):
    def __init__(self, mod_smp_dist, mean_ekin, seed=4096) -> None:
        super().__init__(seed)   
        self.mod_smp_dist = mod_smp_dist
        self.mean_ekin = mean_ekin
        self.z_origin = 0 # sample position
        self.y_monoChopper = self.y_pulseShaping = -17.5
        self.z_pulseShaping = self.z_origin - 25000 - 1300
        self.z_monoChopper = self.z_pulseShaping + 25000

    def makeWorld(self):
        world = Volume("world", Box(5000, 5000, 40000))

        pulseShapingChopperVol = makeDiskChopper(20,15,0,1,731,10)
        monochromaticChopperVol = makeDiskChopper(20,15,0,3, 4629.62962962963,5)

        transformationPS = Transformation3D(0,self.y_pulseShaping,self.z_pulseShaping)
        transformationMono = Transformation3D(0, self.y_monoChopper, self.z_monoChopper)
        world.placeChild('pulseShapingChopperLV', pulseShapingChopperVol, transformationPS)
        world.placeChild('monochromaticChopperLV', monochromaticChopperVol, transformationMono)

        # mat = 'physics=idealElaScat;xs_barn=1;density_per_aa3=5;energy_transfer_eV=0.03'
        mat = "physics=ncrystal;nccfg='bzscope_c1_vol_77K.ncmat;bragg=1';scatter_bias=10.0;abs_bias=1.0;"
        # mat = 'bzscope_c1_vol_77K.ncmat;bragg=0'
        sample = Volume("sample", Sphere(0, 1),matCfg=mat)
        ms = MultiScatCounter()
        ms.make(sample)
        world.placeChild('samplePV', sample, Transformation3D(0,0,self.z_origin))

        mon = Volume("mon", Box(50,50,1e-6))
        tofScorerPS = TOFHelper("tofMonScorerPulseShaping", 0.005440, 0.0055, groupID=1)
        tofScorerPS.make(mon)
        tofScorerMono = TOFHelper("tofMonScorerMono", 0.0, 0.03, groupID=2)
        tofScorerMono.make(mon)

        engScorerBeforeSample = ESpectrumHelper("engMonBeforeSample", 0.01, 0.02,groupID=3)
        engScorerBeforeSample.make(mon)

        world.placeChild("tofMonPSLV", mon, Transformation3D(0,self.y_pulseShaping,self.z_pulseShaping+0.01), scorerGroup=1)
        world.placeChild("tofMonMonoLV", mon, Transformation3D(0, self.y_monoChopper, self.z_monoChopper+0.01), scorerGroup=2)
        world.placeChild("engMonBeforeSampleLV", mon, Transformation3D(0, 0, self.z_origin-5), scorerGroup=3)

        deg = np.pi/180.

        monitor = Volume("mon", Sphere(2999.99, 3000, starttheta=1.*deg, deltatheta=178*deg ))
        # helper = DirectSqwHelper('sqw', self.mod_smp_dist, self.mean_ekin, [0,0,1.], [0,0,0],
        #          qmin = 1e-1, qmax = 12, num_qbin = 500, 
        #          ekinmin=-0.1, ekinmax=0.06,  num_ebin = 500,
        #          pdg = 2112, groupID  = 0, ptstate = 'ENTRY')
        helper = ESpectrumHelper("engScorerDetector", -0.1, 0.1,linear=True)
        helper.make(monitor)
        helper.addScatterCounter(ms, 1)

        world.placeChild('monPV', monitor,Transformation3D(0,0,0))

        self.setWorld(world)


class MyGun(PythonGun):
    def __init__(self, mod_smp_dist, ekin ):
        super().__init__(2112)
        self.ekin = ekin
        self.mod_smp_dist = mod_smp_dist

    def samplePosition(self):
        return 0,0,self.mod_smp_dist
    
    def sampleEnergy(self):
        if isinstance(self.ekin, np.ndarray):
            r=np.random.uniform(self.ekin[0], self.ekin[1], 1)[0]
            return velocity2ekin(r)
        else:
            return self.ekin

mod_smp_dist = 0 - 1300 - 25000 - 10000
ekin = [0.017, 0.018]

const_neutron_mass = 1.674e-27 #kg
const_neutron_mass_amu = 1.00866491588
const_c  = 299792458 # (m/s)/c
const_dalton2eVc2 =  931494095.17
const_neutron_mass_evc2 = const_neutron_mass_amu * const_dalton2eVc2  #[ eV/(Aa/s)^2 ]

def ekin2velocity(ekin):
    return np.sqrt(2 * ekin / (const_neutron_mass_evc2 / const_c**2))

def velocity2ekin(velocity):
    return 0.5 * const_neutron_mass_evc2 / const_c**2  * (velocity)**2

gun = MyGun(mod_smp_dist, ekin2velocity(np.array(ekin)))        # uniform time-of-flight

sim = MySim(mod_smp_dist, 0.017532, seed=1010)
sim.makeWorld()

# sim.show(gun, 100)

sim.simulate(gun, 1e8)
tofPS = sim.gatherHistData("tofMonScorerPulseShaping")
tofMono = sim.gatherHistData("tofMonScorerMono")
engBfSample = sim.gatherHistData("engMonBeforeSample")
engDetector = sim.gatherHistData("engScorerDetector")
if sim.rank==0:
    print('Testing : ',tofMono.getTotalWeight())
    print(f"Energy before sample: {(engBfSample.getEdge()[1:] * engBfSample.getWeight()).sum()/engBfSample.getAccWeight()}")
    engBfSample.savefig('engBfSample.png')
    tofPS.savefig('tofps.png')
    tofMono.savefig('tofmono.png')
    engDetector.savefig('engDetector.png')


# res = sim.gatherHistData('sqw')
# destination = 0
# if sim.rank==destination:
#     # res.plot(False, log=False)
#     # res.plot(True, title=f'Total Weight: {res.getAccWeight()}', log=True)
#     fn = 'Si_bzscope_braggIncluded'
#     plt=res.plot(title=f'{fn}', log=True)
#     plt.xlabel(r'Q($\AA^{-1}$)')
#     plt.ylabel(r'$\omega$(eV)')
#     plt.tight_layout()
#     plt.grid(False)
#     plt.savefig(f'{fn}.pdf')
