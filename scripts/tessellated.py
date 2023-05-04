#!/usr/bin/env python3

from Cinema.Prompt import Launcher, Visualiser
from Cinema.Prompt.solid import Box, Tessellated
from Cinema.Prompt.gun import PythonGun

from Cinema.Prompt.geo import Volume, Transformation3D

gunCfg = "gun=IsotropicGun;energy=0.001;position=0,0,0"
scorerCfg_det = "Scorer=NeutronSq;name=SofQ;sample_position=0,0,1;beam_direction=0,0,1;dist=-100;ptstate=ENTRY;linear=yes;Qmin=0.5; Qmax = 50; numbin=100"
scorerCfg_detpsd = "Scorer=PSD;name=NeutronHistMap;xmin=-500;xmax=500;numBins_x=10;ymin=-500;ymax=500;numBins_y=10;ptstate=SURFACE;type=XY"

# matCfg_sample = "physics=ncrystal;nccfg='LiquidHeavyWaterD2O_T293.6K.ncmat';scatter_bias=10;abs_bias=1"
matCfg_sample = "Cd.ncmat"
l = Launcher()
l.setSeed(4096)
world = Volume("world", Box(600, 600, 600))


import pyvista
sphere = pyvista.Sphere(100)
t = Tessellated(sphere.faces, sphere.points)
tes = Volume('T', t, matCfg = matCfg_sample) 

# tes = Volume('T', Box(10,10,10), matCfg = matCfg_sample) 

world.placeChild("Tessellated_TP", tes, Transformation3D(0., 0., 0))

l.setWorld(world)
l.setGun(gunCfg)
l.showWorld(100)

# gun = PythonGun()
# l.setPythonGun(gun)

# l.go(1000000)

# psd = l.getHist(scorerCfg_detpsd)
# psd.plot(show=True)
