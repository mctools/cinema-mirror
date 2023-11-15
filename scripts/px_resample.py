#!/usr/bin/env python3

import mcpl
import matplotlib.pyplot as plt
import numpy as np
from  Cinema.Prompt.Histogram import Hist2D

def plot(fn, show=False):
    # fn='res.mcpl.gz'
    # N=mcpl.MCPLFile(fn).nparticles

    myfile = mcpl.MCPLFile(fn, 10_000)
    hist_sample=Hist2D(0, 3, 100,
                0, 0.07, 100) 
    
    for p in myfile.particle_blocks:
    #    print(p.ekin, p.x, p.y, p.z, p.weight)
        Q = np.sqrt(p.x**2 + p.y**2 + p.z**2)
        hist_sample.fillmany(Q.astype(float), p.ekin.astype(float),p.weight.astype(float))

    hist_sample.plot(show=show)

fn='samples.mcpl.gz'
plot(fn)
plot('res.mcpl.gz', True)
   
