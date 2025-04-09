#!/usr/bin/env python3

from Cinema.Prompt.scorer import WlSpectrum
import numpy as np

import Cinema.Prompt as cpt

expectWl = [425., 454., 427., 456., 425., 447., 392., 415., 362., 334., 295.,
       276., 271., 234., 243., 194., 192., 183., 157., 156.]

def testgun():
    gunCfg = "gun=MaxwellianGun;src_w=2;src_h=2;src_z=-100;slit_w=2;slit_h=2;slit_z=1e99;temperature=293;"
    return gunCfg

def nc_cfgs():
    cfgs = [
        "Al_sg225.ncmat",
        "physics=ncrystal;nccfg='Al_sg225.ncmat';scatter_bias=2.0;abs_bias=1.0"
    ]
    return cfgs

class MySim(cpt.Prompt):
    def __init__(self, seed, cfg) -> None:
        super().__init__(seed)
        self.sample = cfg

    def makeWorld(self):
        world = cpt.geo.Volume('world', cpt.solid.Box(50, 50, 200))

        hx = 1
        hy = 1
        hz = 10

        sample = cpt.geo.Volume('sample', cpt.solid.Box(hx, hy, hz), self.sample)
        world.placeChild('entity', sample)

        dtt = cpt.geo.Volume('detector', cpt.solid.Box(10, 10, 1))
        scorerWl = WlSpectrum()
        scorerWl.cfg_name = 'WavelengthSp'
        scorerWl.cfg_min = 1
        scorerWl.cfg_max = 2
        scorerWl.cfg_numbin = 20
        dtt.addScorer(scorerWl)
        world.placeChild('detectorPhy', dtt, cpt.geo.Transformation3D(0,0,90))

        self.setWorld(world)

def build(cfg):
    sim = MySim(seed=4096, cfg=cfg)
    sim.makeWorld()
    return sim

def run(sim, nparticles):
    sim.simulate(testgun(), nparticles)

def viz(sim):
    sim.show(testgun(), 100)

if __name__ == "__main__":
    nparticles = 1e6
    sim = build(nc_cfgs()[1])
    # viz(sim)
    run(sim, nparticles)
    wlhist = sim.gatherHistData('WavelengthSp')
    # wlhist.plot(1)
