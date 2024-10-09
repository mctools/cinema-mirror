#!/usr/bin/env python3

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.component import DiskChopper
from Cinema.Prompt.solid import Box, Sphere, Tube
from Cinema.Prompt.scorer import VolFluenceHelper, ESpectrumHelper, DepositionHelper
from Cinema.Prompt.physics import Material
from Cinema.Prompt.gun import IsotropicGun, SimpleThermalGun, PythonGun
from Cinema.Prompt.GidiSetting import GidiSetting 
import matplotlib.pyplot as plt
from Cinema.Interface import plotStyle
import numpy as np
import openmc
import os
os.environ['OPENMC_CROSS_SECTIONS']='/home/xxcai1/git/openmc/data/endfb-viii.0-hdf5/cross_sections.xml'

# export PATH="/home/caixx/git/openmc/build/bin/:$PATH" 

plotStyle()

cdata=GidiSetting()
cdata.setGidiThreshold(5)
cdata.setEnableGidi(True)
cdata.setEnableGidiPowerIteration(False)
cdata.setGammaTransport(True)
# energy = [20e3, 149e3]  #U238 URR

# U235, URR_domainMin 0.002251MeV,  URR_domainMax 0.024999MeV.
# energy = [.002251e6, .024999e6]

# energy = [1e6, 6e6]
# energy = [1e5, 10e5]
# energy = [1e4, 10e4]
# energy = [1e3, 10e3]
# energy = [1e2, 10e2]
# energy = [1e1, 10e1]
# energy = [1, 10]
# energy = [1e-1, 10e-1]
energy=1e6

partnum = 1e4
loweredge=1e-5
upperedge=70e6

numbin_en=300
numbin_mu=30
radius_mm = 1e-5
hlen_mm = 1e20
# #########################################################

cfg='freegas::Th/18gcm3'

# cfg='freegas::Si/18gcm3'

# cfg='freegas::U/18.8gcm3/U_is_0.3000_U238_0.7000_U235;temp=293.6'
# cfg='freegas::H2O/1gcm3/H_is_H1/O_is_O16;temp=293.6'
# cfg='freegas::H/1gcm3/H_is_H1;temp=293.6'
# cfg='freegas::O/1gcm3/O_is_O16'

# cfg='freegas::C/18gcm3/C_is_C13'
# cfg='freegas::Bi/18gcm3'
# cfg='freegas::U/1.8gcm3/U_is_U238'
# cfg='freegas::U/1.8gcm3/U_is_U235'
# cfg='freegas::U/1.8gcm3/U_is_U233'
# cfg = 'freegas::U/18.8gcm3/U_is_0.1000_U238_0.9000_U235;temp=293.6'
# cfg='freegas::Ag/18gcm3'

# cfg='Al_sg225.ncmat'
# cfg='LiquidWaterH2O_T293.6K.ncmat;density=1gcm3;temp=293.6'
# cfg='freegas::He/1.8e-3gcm3/He_is_He3'

# cfg='freegas::B/18gcm3'

class MySim(Prompt):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)   

    def makeWorld(self):
        world = Volume("world", Tube(0, hlen_mm + 1e-3, hlen_mm))
        lw = Material(cfg) 
        media = Volume("media", Tube(0, hlen_mm, hlen_mm), matCfg= lw)
        world.placeChild('media', media)

        VolFluenceHelper('volFlux', min=loweredge, max=upperedge, numbin=numbin_en, pdg=22).make(media)
        # ESpectrumHelper('ESpec', min=loweredge, max=upperedge, numbin=numbin_en, ptstate='PEA_POST', pdg=22).make(media)
        self.setWorld(world)

sim = MySim(seed=1010)
sim.makeWorld()

class MyGun(PythonGun):
    def __init__(self, pdg, ekin):
        super().__init__(pdg)
        self.ekin = ekin

    def samplePosition(self):
        return 0,0,0.
    
    def sampleEnergy(self):
        if isinstance(self.ekin, list):
            r=np.random.uniform(self.ekin[0], self.ekin[1], 1)[0]
            return r
        else:
            return self.ekin


gun = MyGun(int(22), energy)

# gun = SimpleThermalGun()
# gun.setEnergy(energy)
# gun.setPosition([0,0,-1e5])

# sim.show(gun, 1)
# vis or production
sim.simulate(gun, partnum)


###############################################################################################
# openmc

import openmc
from openmc.data import K_BOLTZMANN, NEUTRON_MASS
import numpy as np
import matplotlib.pyplot as plt

# export PATH="/home/caixx/git/openmc/build/bin:$PATH"
# (cinemavirenv) caixx@x:~/git/cinema/rundir/del$ export OPENMC_CROSS_SECTIONS=~/git/openmc/build/data/endfb-viii.0-hdf5/cross_sections.xml


def run(energy, numPart):
    """Generate the OpenMC input XML

    """
    # Define material
    import openmc
    mat = openmc.Material.from_ncrystal(cfg)
    print(mat)
    import openmc
    import numpy as np

    # Define materials
    materials = openmc.Materials()
    materials.append(mat)
    materials.export_to_xml()


    # Set up geometry
    x1 = openmc.XPlane(x0=-1.e9, boundary_type='reflective')
    x2 = openmc.XPlane(x0=+1.e9, boundary_type='reflective')
    y1 = openmc.YPlane(y0=-1.e9, boundary_type='reflective')
    y2 = openmc.YPlane(y0=+1.e9, boundary_type='reflective')
    z1 = openmc.ZPlane(z0=-1.e9, boundary_type='reflective')
    z2 = openmc.ZPlane(z0=+1.e9, boundary_type='reflective')
    cell = openmc.Cell(fill=materials)
    cell.region = +x1 & -x2 & +y1 & -y2 & +z1 & -z2
    geometry = openmc.Geometry([cell])
    geometry.export_to_xml('geometry.xml')

    # Define source
    source = openmc.Source()
    source.space = openmc.stats.Point((0,0,0))
    source.angle = openmc.stats.Isotropic()
    if isinstance(energy, list):
        source.energy = openmc.stats.Uniform(energy[0], energy[1])
    else:
        source.energy = openmc.stats.Discrete([energy], [1.])


    # Settings
    settings = openmc.Settings()
    settings.source = source
    settings.particles = numPart
    settings.run_mode = 'fixed source'
    settings.batches = 1
    settings.create_fission_neutrons = False
    settings.export_to_xml('settings.xml')

    # Define tallies
    energy_bins = np.logspace(np.log10(loweredge), np.log10(upperedge), numbin_en + 1)
    energy_filter = openmc.EnergyFilter(energy_bins)
    tally = openmc.Tally(name='tally')
    tally.filters = [energy_filter]
    tally.scores = ['flux']
    tallies = openmc.Tallies([tally])
    tallies.export_to_xml('tallies.xml')


    openmc.run(tracks=False)


def read_results(filename, tally, filter):
    with openmc.StatePoint(filename) as sp:        
        t = sp.get_tally(name=tally)
        x = 0.5*(t.find_filter(filter).bins[:,1]+t.find_filter(filter).bins[:,0])

        # t = sp.get_tally(name='tally')
        # x = 0.5*(t.find_filter(openmc.EnergyFilter).bins[:,1]+t.find_filter(openmc.EnergyFilter).bins[:,0])

        y = t.mean[:,0,0]
        std_dev = t.std_dev[:,0,0]

    return x, y, std_dev


# def plot():
#     """Extract and plot the results

#     """
#     # Read results
#     path =  f'statepoint.1.h5'
#     x1, y1, sd = read_results(path)

#     # y1 /= np.diff(x1)*sum(y1)
#     print('y sum()', y1.sum())

#     # Set up the figure
#     # plt.figure(1, facecolor='w', figsize=(8,8))
#     plt.plot(x1, y1*partnum,'-o', label=f'openmc {y1.sum()*partnum}')
#     plt.legend(loc=0)
#     plt.show()

run(energy, int(partnum))


#################################################################################################################################
# import os
# os.system(f'openmc' )
# plot()

sp = 'statepoint.1.h5' 

##########################################################
# energy
plt.figure()
sim.gatherHistData('volFlux').plot(show=False, log=True, label='Prompt  ', title='Energy distribution')
x, y, std = read_results(sp, 'tally', openmc.EnergyFilter) 
plt.plot(x, y*partnum,'-o', label=f'Openmc {y.sum()*partnum}')
plt.legend(loc=0)




plt.show()

