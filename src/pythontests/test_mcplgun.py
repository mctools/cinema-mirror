#!/usr/bin/env python3

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box,Tube
from Cinema.Prompt.scorer import makePSD, ESpectrumHelper, WlSpectrumHelper, TOFHelper, VolFluenceHelper
from Cinema.Prompt.gun import PythonGun
from Cinema.Prompt.histogram import wl2ekin
from Cinema.Prompt.physics import Material, Mirror
from Cinema.Prompt.component import makeTrapezoidGuide
from Cinema.Prompt.gun import UniModeratorGun, MCPLGun

import numpy as np

from Cinema.Prompt.files import MCPLParticle, MCPLBinaryWrite

par = MCPLParticle(0.0253,  2,3,4,  5,6,7., 0.,1., 0)
wrt = MCPLBinaryWrite('test.mcpl')
wrt.write(par)
par.direction_x = 1.
par.direction_y = 0.
par.direction_z = 0.
wrt.write(par)

del wrt

import mcpl
myfile = mcpl.MCPLFile("test.mcpl.gz", blocklength=10)
print(myfile.nparticles)
for p in myfile.particle_blocks:
   print( p.ekin, p.polarisation, p.direction )


# # Reproduction of the figure 13 of the paper https://doi.org/10.1016/j.cpc.2023.109004

# class MySim(PromptMPI):
#     def __init__(self, seed=4096) -> None:
#         super().__init__(seed)   

#     def makeWorld(self):
#         universe = Material('freegas::H1/1e-26kgm3')
#         universe.setBiasScat(1.)
#         universe.setBiasAbsp(1.)

#         world = Volume("world", Box(30000, 30000, 30000), matCfg=universe)

#         # detector1 = Volume("det1", Box(25, 25, 0.0001))
#         # makePSD('psd1', detector1, 20, 20 )
#         # world.placeChild("det", detector1, Transformation3D(0., 0., -1000))

#         # detector2 = Volume("det2", Box(35, 35, 0.0001))
#         # makePSD('psd2', detector2, 20, 20 )
#         # world.placeChild("det2", detepctor2, Transformation3D(0., 0., 1200))

#         # ESpectrumHelper('espec').make(detector2)
#         # WlSpectrumHelper('wlspec').make(detector2)
#         # TOFHelper('tof').make(detector2)

#         # guide = makeTrapezoidGuide(500, 25,25,25,25, 1.0, 10)
#         # world.placeChild("guide", guide, Transformation3D(0., 0., -400))

#         self.setWorld(world)



# sim = MySim(seed=1010)
# sim.makeWorld()


# gun = MCPLGun('samples.mcpl.gz')
# # gun = MCPLGun('source.xml')

# # gun = UniModeratorGun()
# # gun.setWlMean(3.39)
# # gun.setWlRange(0.3)
# # gun.setSlit([50,50,-1000])
# # gun.setSource([50,50,-2000])



# # vis or production
# if False:
#     sim.show(gun, 1)
# else:
#     sim.simulate(gun, 1e6)

# # destination = 0
# # psd1 = sim.gatherHistData('psd1', dst=destination)
# # psd2 = sim.gatherHistData('psd2', dst=destination)
# # wlspec = sim.gatherHistData('wlspec', dst=destination)
# # if sim.rank==destination:
# #     psd1.plot(show=True, log=False)
# #     psd2.plot(show=True, log=False)
# #     wlspec.plot(show=True, log=False)
