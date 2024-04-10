#!/usr/bin/env python3

from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.solid import Box, Sphere, Tube
from Cinema.Prompt.scorer import VolFluenceHelper, ESpectrumHelper
from Cinema.Prompt.physics import Material
from Cinema.Prompt.gun import IsotropicGun
from Cinema.Prompt.centralData import CentralData 
import matplotlib.pyplot as plt

cdata=CentralData()
cdata.setEnableGidi(True)
cdata.setEnableGidiPowerIteration(True)
cdata.setGidiThreshold(4)

class MySim(PromptMPI):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)   

    def makeWorld(self):

        boxhsize = 25.4 # 1 inch
        hlength = 1e50
        reflhthick = 10

        problemlist = []
        # fuel rod radius and composition 
        problemlist.append([12.70, 'freegas::U/18.8gcm3/U_is_0.9902_U238_0.0098_U235;temp=293.6'])
        problemlist.append([6.350, 'freegas::U/18.8gcm3/U_is_0.9650_U238_0.0350_U235;temp=293.6'])
        problemlist.append([3.175, 'freegas::U/18.8gcm3/U_is_0.3000_U238_0.7000_U235;temp=293.6'])

        idx = 0
        fradius = problemlist[idx][0]
        lw = Material('freegas::H2O/1gcm3/H_is_H1/O_is_O16;temp=293.6') 
        # lw = Material('LiquidWaterH2O_T293.6K.ncmat;density=1gcm3;temp=293.6')      

        fuel = Material(problemlist[idx][1]) 

        world = Volume("world", Box(boxhsize + reflhthick*2, boxhsize + reflhthick*2, hlength + reflhthick*2), matCfg=lw)

        refboudary3rd = Volume("refboudary3rd", Box(boxhsize, boxhsize, reflhthick), surfaceCfg=f'physics=Mirror;m=-1')
        world.placeChild('refboudary1', refboudary3rd, transf=Transformation3D(0,0,-hlength-reflhthick))
        world.placeChild('refboudary2', refboudary3rd, transf=Transformation3D(0,0,hlength+reflhthick))

        refboudary = Volume("refboudary", Box(boxhsize, reflhthick, hlength), surfaceCfg=f'physics=Mirror;m=-1')
        world.placeChild('refboudary3', refboudary, transf=Transformation3D(0,-boxhsize-reflhthick,0))
        world.placeChild('refboudary4', refboudary, transf=Transformation3D(0,boxhsize+reflhthick,0))
        world.placeChild('refboudary5', refboudary, transf=Transformation3D(-boxhsize-reflhthick,0,0).applyRotZ(90))
        world.placeChild('refboudary6', refboudary, transf=Transformation3D(boxhsize+reflhthick,0,0).applyRotZ(90))

        frod = Volume('fuel', Tube(0, fradius, hlength), matCfg=fuel)
        world.placeChild('frod', frod)


        # # lw = Material('freegas::He/1gcm3/He_is_1_He3') 
        # lw = Material('freegas::B/1gcm3/B_is_1_B10') 
        # # lw = Material('freegas::Li/1gcm3/Li_is_1_Li6') 
        # # lw = Material('freegas::H2O/1gcm3/H_is_1_H1/O_is_1_O16') 

        # lw.setBiasScat(1.)
        # lw.setBiasAbsp(1)
        # sphere = Volume("sphere", Sphere(0, 2), matCfg=lw, surfaceCfg=f'physics=Mirror;m=-1')
        # world.placeChild('sphere', sphere)

        # VolFluenceHelper('spct', max=20e6, numbin=300).make(sphere)
        # ESpectrumHelper('escap', min=1e-5, max=20e6, ptstate='EXIT').make(sphere)
     

        self.setWorld(world)

sim = MySim(seed=1010)
sim.makeWorld()

gun = IsotropicGun()
gun.setEnergy(1e6)
gun.setPosition([0,0,0])

batchsize = 1e4
# vis or production

# sim.show(gun, 1)

sim.simulate(gun, batchsize)

totCycle = 250
settleCycle = 50
totneutron = 0

for i in range(10000):
    num = sim.simulateSecondStack(batchsize)
    print(f'iteration {i}')
    if i>=settleCycle:
        totneutron += num
        print('keff', totneutron/(i-settleCycle+1)/batchsize)


print('keff', totneutron/(totCycle-settleCycle+1)/batchsize)




# destination = 0
# spct = sim.gatherHistData('spct', dst=destination)
# escap= sim.gatherHistData('escap', dst=destination)
# if sim.rank==destination:
#     # escap.plot(log=True)
#     # plt.legend(loc=0)
#     # plt.show()
#     spct.plot(show=True, log=True)
