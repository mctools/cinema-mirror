#!/usr/bin/env python3

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume
from Cinema.Prompt.solid import Box, Sphere, Tube
from Cinema.Prompt.scorer import VolFluenceHelper, ESpectrumHelper, DepositionHelper
from Cinema.Prompt.physics import Material
from Cinema.Prompt.gun import IsotropicGun, SimpleThermalGun, PythonGun
from Cinema.Prompt.GidiSetting import GidiSetting 
import matplotlib.pyplot as plt
from Cinema.Interface import plotStyle
import numpy as np

import os
os.environ['OPENMC_CROSS_SECTIONS']='/home/caixx/git/openmc/data/endfb-viii.0-hdf5/cross_sections.xml'

# export PATH="/home/caixx/git/openmc/build/bin/:$PATH" 

plotStyle()

cdata=GidiSetting()
cdata.setGidiThreshold(5)
cdata.setEnableGidi(True)
cdata.setEnableGidiPowerIteration(False)

# energy = [5e6, 10e6]

energy=1e6

partnum = 1e4
loweredge=1e-5
upperedge=70e6

numbin_en=300
numbin_mu=100
radius_mm = 1e-4

#########################################################


nc_bxs=80.27
endf_fxs=20.43608

const_neutron_mass_amu =1.00866491588
h1_mass =               1.00782503224
awr=h1_mass/(h1_mass+const_neutron_mass_amu)

nc_fxs = nc_bxs*awr*awr
endf_bxs=endf_fxs/awr/awr


###############################################################


# cfg='freegas::Th/18gcm3'

# cfg='freegas::Si/18gcm3'

# cfg='freegas::U/18.8gcm3/U_is_0.3000_U238_0.7000_U235;temp=293.6'
# cfg='freegas::H2O/1gcm3/H_is_H1/O_is_O16;temp=293.6'
# cfg='freegas::H/1gcm3/H_is_H1;temp=293.6'
# cfg='freegas::O/1gcm3/O_is_O16'

cfg='freegas::U/1.8gcm3/U_is_U235'
# cfg='freegas::C/18gcm3/C_is_C13'
# cfg='freegas::Bi/18gcm3'
# cfg='freegas::U/18gcm3/U_is_U238'


# cfg='Al_sg225.ncmat'
# cfg='LiquidWaterH2O_T293.6K.ncmat;density=1gcm3;temp=293.6'

class MySim(Prompt):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)   

    def makeWorld(self):
        size = 1e-3

        world = Volume("world", Tube(0, size, 1.1e50))
        # lw = Material('freegas::He/1gcm3/He_is_1_He3') 
        # lw = Material('freegas::B/1gcm3/B_is_1_B10') 
        # lw = Material('freegas::Li/1gcm3/Li_is_1_Li6') 
        # lw = Material('freegas::H/1gcm3/H_is_H1') 
        lw = Material(cfg) 
        # lw = Material('freegas::O/1gcm3/O_is_O16') 
        lw.setBiasAbsp(1)
        lw.setBiasScat(1)
        media = Volume("media", Tube(0, radius_mm, 1e50), matCfg= lw)
        world.placeChild('media', media)

        # VolFluenceHelper('volFlux', max=20e6, numbin=300).make(media)
        ESpectrumHelper('ESpec', min=loweredge, max=upperedge, numbin=numbin_en, ptstate='EXIT').make(media)
        media.addScorer(f'Scorer=Angular;name=SofAngle;sample_pos=0,0,1;beam_dir=0,0,1;dist=-100;ptstate=EXIT;linear=yes;min=-1;max=1;numbin={numbin_mu}')

        DepositionHelper('dep', pdg=2112, min=loweredge, max=upperedge, numbin=numbin_en, ptstate='PROPAGATE_POST', linear=False).make(media)
        self.setWorld(world)



sim = MySim(seed=1010)
sim.makeWorld()

class MyGun(PythonGun):
    def __init__(self, pdg, ekin):
        super().__init__(pdg)
        self.ekin = ekin

    def samplePosition(self):
        return 0,0,-1.e5
    
    def sampleEnergy(self):
        if isinstance(self.ekin, list):
            r=np.random.uniform(self.ekin[0], self.ekin[1], 1)[0]
            return r
        else:
            return self.ekin


gun = MyGun(int(2112), energy)

# gun = SimpleThermalGun()
# gun.setEnergy(energy)
# gun.setPosition([0,0,-1e50])


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
    mat = openmc.Material.from_ncrystal(cfg)

    # mat = openmc.Material(1, "h")
    # # Add nuclides to uo2
    # mat.add_nuclide('H1', 1.0)
    # mat.set_density('g/cm3', 1.0)

    print(mat)
    materials = openmc.Materials()
    materials.append(mat)
    materials.export_to_xml()


    # Instantiate surfaces
    cyl = openmc.XCylinder(boundary_type='vacuum', r=radius_mm*0.1)
    px1 = openmc.XPlane(boundary_type='vacuum', x0=-1.)
    px2 = openmc.XPlane(boundary_type='transmission', x0=1.)
    px3 = openmc.XPlane(boundary_type='vacuum', x0=1.e9)

    # Instantiate cells
    inner_cyl_left = openmc.Cell()
    inner_cyl_right = openmc.Cell()
    outer_cyl = openmc.Cell()

    # Set cells regions and materials
    inner_cyl_left.region = -cyl & +px1 & -px2
    inner_cyl_right.region = -cyl & +px2 & -px3
    outer_cyl.region = ~(-cyl & +px1 & -px3)
    inner_cyl_right.fill = mat

    # Create root universe and export to XML
    geometry = openmc.Geometry([inner_cyl_left, inner_cyl_right, outer_cyl])
    geometry.export_to_xml('geometry.xml')

    # Define source
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point((0,0,0))
    source.angle = openmc.stats.Monodirectional()
    
    if isinstance(energy, list):
        source.energy = openmc.stats.Uniform(energy[0], energy[1])
    else:
        source.energy = openmc.stats.Discrete([energy], [1.])
    
    source.particle = 'neutron'

    # Settings
    settings = openmc.Settings()
    # settings.temperature = 293.6
    settings.source = source
    settings.particles = numPart
    settings.run_mode = 'fixed source'
    settings.batches = 1
    settings.photon_transport = False
    cutoff=0.05
    settings.cutoff = {'energy_photon' : cutoff}
    settings.export_to_xml( 'settings.xml')
    settings.max_tracks = settings.particles
    


    # Define filters
    surface_filter = openmc.SurfaceFilter(cyl)
    particle_filter = openmc.ParticleFilter('neutron')
    energy_bins = np.logspace(np.log10(loweredge),
                                np.log10(upperedge), numbin_en+1)
    energy_filter = openmc.EnergyFilter(energy_bins)

    # Create tallies and export to XML
    tally = openmc.Tally(name='tally')
    tally.filters = [surface_filter, energy_filter, particle_filter]
    tally.scores = ['current']


    mufilter = openmc.MuFilter(numbin_mu)
    tally2 = openmc.Tally(name='tally2')
    tally2.filters = [ surface_filter, mufilter, particle_filter]
    tally2.scores = ['current']

    cellfilter = openmc.CellFilter(inner_cyl_right)
    tally3 = openmc.Tally(name='tally3')
    tally3.filters = [ energy_filter, cellfilter, particle_filter ]
    tally3.scores = ['flux']

    tallies = openmc.Tallies([tally, tally2, tally3])
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
# mu
sim.gatherHistData('SofAngle').plot(show=False, log=False)

x, y, std = read_results(sp, 'tally2', openmc.MuFilter) 
plt.plot(x, y*partnum,'-o', label=f'openmc {y.sum()*partnum}')
plt.legend(loc=0)


##########################################################
# energy
plt.figure()
sim.gatherHistData('ESpec').plot(show=False, log=True)
x, y, std = read_results(sp, 'tally', openmc.EnergyFilter) 
plt.plot(x, y*partnum,'-o', label=f'openmc {y.sum()*partnum}')
plt.legend(loc=0)


plt.figure()
sim.gatherHistData('dep').plot(show=False, log=True)
x, y, std = read_results(sp, 'tally3', openmc.EnergyFilter) 
plt.plot(x, y*partnum,'-o', label=f'openmc {y.sum()*partnum}')
plt.legend(loc=0)


plt.show()

