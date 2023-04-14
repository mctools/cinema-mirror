#!/usr/bin/env python3

################################################################################
##                                                                            ##
##  This file is part of Prompt (see https://gitlab.com/xxcai1/Prompt)        ##
##                                                                            ##
##  Copyright 2021-2022 Prompt developers                                     ##
##                                                                            ##
##  Licensed under the Apache License, Version 2.0 (the "License");           ##
##  you may not use this file except in compliance with the License.          ##
##  You may obtain a copy of the License at                                   ##
##                                                                            ##
##      http://www.apache.org/licenses/LICENSE-2.0                            ##
##                                                                            ##
##  Unless required by applicable law or agreed to in writing, software       ##
##  distributed under the License is distributed on an "AS IS" BASIS,         ##
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  ##
##  See the License for the specific language governing permissions and       ##
##  limitations under the License.                                            ##
##                                                                            ##
################################################################################



from Cinema.Prompt import Prompt, PromptMPI
from Cinema.Prompt.solid import Box, Tube, Trapezoid
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.component import makeTrapezoidGuide
from Cinema.Prompt.gun import PythonGun
import numpy as np



class MySim(PromptMPI):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)

    def setScorer(self):
        self.scorer['PSD1'] = 'Scorer=PSD;name=NeutronHistMap;xmin=-25;xmax=25;numbin_x=20;ymin=-25;ymax=25;numbin_y=20;ptstate=SURFACE;type=XY'
        self.scorer['PSD2'] = 'Scorer=PSD;name=NeutronHistMap2;xmin=-35;xmax=35;numbin_x=20;ymin=-35;ymax=35;numbin_y=20;ptstate=SURFACE;type=XY'


    def makeWorld(self, anyUserParameters=np.zeros(2)):

        air = "freegas::N78O22/1.225kgm3"
        worldsize = 6500.
        world = Volume("world", Box(worldsize*0.5, worldsize*0.5, worldsize*0.5))

        det1 = Volume("detector", Box(50, 50, 0.01), scorerCfg=self.scorer['PSD1'] )
        world.placeChild('det1', det1, Transformation3D(0, 0, 1000))

        world.placeChild('guide', makeTrapezoidGuide(500., 25, 25, 25, 25, 1.), Transformation3D(0, 0, 1600))
     
        det2 = Volume("detector", Box(50, 50, 0.01), scorerCfg=self.scorer['PSD2'] )
        world.placeChild('det2', det2, Transformation3D(0, 0, 3200))

        return world


sim = MySim()

# set gun
gunCfg = "gun=UniModeratorGun;src_w=50;src_h=50;src_z=0;slit_w=50;slit_h=50;slit_z=1100;mean_wl=3.39"
sim.setGun(gunCfg)

# vis or production
if False:
    sim.show(10)
else:
    sim.simulate(1e6)

hist1 = sim.getScorerHist(sim.scorer['PSD1'])
hist2 = sim.getScorerHist(sim.scorer['PSD2'])
# hist1.plot(show=False)
print(f'weight {hist2.getWeight().sum()}')
if sim.rank==0:
    hist2.plot(show=True)