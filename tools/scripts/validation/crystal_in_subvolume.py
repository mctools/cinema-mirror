#!/usr/bin/env python3

# This script test whether single crystal placed inside subvolume (or sub-subvolume ...)
# produces correct responses. 
# It shows that the reflection direction does not depend on subvolume orientation. 

from Cinema.Prompt import Prompt, PromptMPI, Optimiser
from Cinema.Prompt.solid import Box, Tube, Trapezoid, Sphere
from Cinema.Prompt.geo import Volume, Transformation3D
from Cinema.Prompt.scorer import makeFlatPSD

class Sim(PromptMPI):
    def __init__(self, seed=4096) -> None:
        super().__init__(seed)

    def makeWorld(self):
        world = Volume("world", Box(20, 20, 40))
        # gunCfg = "gun=MaxwellianGun;src_w=40;src_h=40;src_z=0;slit_w=40;slit_h=40;slit_z=1e99;temperature=293;"
        gunCfg = "gun=UniModeratorGun;mean_wl=5;range_wl=2.0;src_w=1;src_h=1;src_z=-10;slit_w=1;slit_h=1;slit_z=1e99"
        # gunCfg = "gun=SimpleThermalGun;energy=0.000002045105;position=0,0,-1500;direction=0,0,1"
        # gunCfg = "gun=IsotropicGun;energy=0.1;position=0,0,17000"
        self.setGun(gunCfg)

        matCrystal = "physics=ncrystal;nccfg='C_sg194_pyrolytic_graphite.ncmat;mos=0.1deg;dir1=@crys_hkl:0,0,2@lab:0,0,1;dir2=@crys_hkl:1,0,0@lab:1,0,0;lcaxis=0,0,1;dcutoff=3;inelas=false'"
        crystal = Volume("crystal", Box(6, 6, 1), matCrystal)

        subvol = Volume("subVol", Box(8, 8, 8))
        if False:
            world.placeChild("phyCrystal", crystal, Transformation3D(0, 0, 12).applyRotX(45))
        else:
            angle = -30
            subvol.placeChild("phyCrystal", crystal, Transformation3D().applyRotX(45-angle))
            world.placeChild("phySubvol", subvol, Transformation3D(0, 0, 12).applyRotX(angle))

        self.setWorld(world)

MySim = Sim()
MySim.makeWorld()
MySim.show(100)
