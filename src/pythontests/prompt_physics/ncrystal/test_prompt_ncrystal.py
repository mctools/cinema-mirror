#!/usr/bin/env python3

from Cinema.Prompt.scorer import WlSpectrum
import numpy as np

import Cinema.Prompt as cpt

RANDOM_CHECK = [35403.0, 36580.0, 37471.0, 37283.0, 
                36366.0, 35326.0, 33804.0, 32120.0, 
                29983.0, 28137.0, 26002.0, 24110.0, 
                22425.0, 20868.0, 19356.0, 17658.0, 
                16183.0, 15168.0, 13684.0, 12698.0]


def testgun():
    gunCfg = "gun=MaxwellianGun;src_w=2;src_h=2;src_z=-100;slit_w=2;slit_h=2;slit_z=1e99;temperature=293;"
    return gunCfg

def nc_cfgs():
    cfgs = [
        "Al_sg225.ncmat", # pure NC kernel
        # "physics=ncrystal;nccfg='Al_sg225.ncmat';scatter_bias=1.0;abs_bias=1.0" # should be the same as above when NC changes
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

def run(sim : MySim, nparticles):
    sim.simulate(testgun(), nparticles)

def viz(sim : MySim):
    sim.show(testgun(), 100)

def result_plot(sim : MySim):
    wlhist = sim.gatherHistData('WavelengthSp')
    wlhist.plot(1)
    return wlhist

def test_case_restrict(sim : MySim): # This one might be too restrictive
    wlhist = sim.gatherHistData('WavelengthSp')
    # print(wlhist.getHit().tolist(), sep=',')
    np.testing.assert_array_equal(wlhist.getHit(), RANDOM_CHECK)



if __name__ == "__main__":
    nparticles = 1e6
    sim = build(nc_cfgs()[0])
    # viz(sim)
    run(sim, nparticles)
    test_case_restrict(sim)

