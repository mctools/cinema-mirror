#!/usr/bin/env python3

from Cinema.Prompt import Prompt
from Cinema.Prompt.geo import Volume
from Cinema.Prompt.solid import Box
from Cinema.Prompt.gun import PythonGun
from Cinema.Prompt.scorer import VolFluenceHelper
import numpy as np

expVolumicFluence = [0.0, 0.0, 0.013447707964539776, 0.015589563257437062, 0.10172960912641055, 0.699787016199494, 3.1309556996150443, 12.325617113789392, 17.22856163362044, 0.9692521318287943]

class MySim(Prompt):
    def __init__(self, seed) -> None:
        super().__init__(seed)

    def makeWorld(self):
        world = Volume('world', Box(50, 50, 200))
        
        sample_mat = "physics=idealElaScat;xs_barn=5;density_per_aa3=0.5;energy_transfer_eV=0.01"
        dtt = Volume('detector', Box(10, 10, 1), sample_mat)
        scorertof = VolFluenceHelper('testObj', 1e-6, 0.5, 10, 2112, groupID=1)
        scorertof.make(dtt)

        world.placeChild('detectorPhy', dtt, scorerGroup=1)

        self.setWorld(world)

sim = MySim(seed=4096)
sim.makeWorld()
gunCfg = "gun=MaxwellianGun;src_w=2;src_h=2;src_z=-100;slit_w=2;slit_h=2;slit_z=1e99;temperature=293;"
# gunCfg = "gun=UniModeratorGun;mean_wl=1;range_wl=0.1;src_w=2;src_h=2;src_z=-100;slit_w=2;slit_h=2;slit_z=1e99"
# sim.show(gunCfg, 100)
sim.simulate(gunCfg, 1e4)
wlhist = sim.gatherHistData('testObj')
print(list(wlhist.getWeight()), sep=',')
np.testing.assert_allclose(wlhist.getWeight(), expVolumicFluence)

