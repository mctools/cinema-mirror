#!/usr/bin/env python3

# /home/xxcai1/git/cinema/external/KDSource/install/bin/mcpltool -m merged test_killermcpl*
# pymcpltool --stats --pdf merged.mcpl

import numpy as np
from Cinema.PiXiu.io.cell import QeXmlCell
import phonopy
import matplotlib.pyplot as plt
import mcpl, os

np.set_printoptions(suppress=True)
from Cinema.PiXiu.phon import CohPhon

# rundir ~/git/cinema/rundir/px/newNC/si

resN = 1e7
Qmesh = 100
Qrange = 3
# ## Data generation
# Qx, Qy, Qz = (50, 50, 50)
# x = np.linspace(0, 10, Qx)
# y = np.linspace(0, 10, Qy)
# z = np.linspace(0, 10, Qz)

# xv, yv, zv = np.meshgrid(x, y, z)
# Qpoints = np.stack((xv.flatten(), yv.flatten(), zv.flatten()), axis=-1)

Qpoints = np.random.rand(Qmesh**3, 3)*Qrange
ph = CohPhon(temperature=30)
Qmag, en, Smag = ph.s(Qpoints)

# plt.figure()
# plt.hist(en.flatten(), bins=100)
# plt.figure()
# plt.hist(np.log(Smag.flatten()), range=[-200, 1], bins=100,log=True)
# plt.show()

# print(Smag)


##  save to ssv
import kdsource as kds
pt = "n" 

ssvfile = "samples.ssv"

Es = np.atleast_2d(en.flatten()).T
ws = Smag.flatten()

# ws = np.ones_like(ws)
# ws = np.atleast_2d(np.random.random(ws.size)).T

N = Es.shape[0]
dirs = np.zeros((N,3))
dirs[:, 0] = 1
poss = np.repeat(Qpoints, 6, axis=0)

# Time
ts = np.zeros((N,1))
# Stack energies, positions, directions and times
print('parts shape', Es.shape, poss.shape, dirs.shape, ts.shape)
parts = np.concatenate((Es,poss,dirs,ts), axis=1)
np.random.shuffle(parts)
kds.savessv(pt, parts, ws, ssvfile)


os.system("/home/caixx/git/cinema/external/KDSource/bin/ssv2mcpl samples.ssv samples.mcpl")
os.system('pymcpltool --stats --pdf samples.mcpl.gz')
os.system('mv mcpl.pdf mcpl_org.pdf')


# fitting and resample
#######################################################################################################################
import os
import numpy as np
import kdsource as kds
import mcpl

samples = "samples.mcpl.gz"
N=mcpl.MCPLFile(samples).nparticles

# PList: wrapper for MCPL file
plist = kds.PList(samples)
# Geometry: define metrics for variables
geom = kds.Geometry([kds.geom.Energy(),
                     kds.geom.Vol(),
                     kds.geom.Polar()])
# Create KDSource
s = kds.KDSource(plist, geom, kernel='epa')

# Give a little more importance to energy
var_importance = [3,1,1,1,1,1]

parts,ws = s.plist.get(N=-1)
scaling = s.geom.std(parts=parts)
scaling /= var_importance


# Method 3: Adaptive Maximum Likelihood Cross-Validation:
# Creates a grid of adaptive bandwidths and evaluates the
# cross-validation scores on each one, which is an indicator of the
# quality of the estimation. Selects the bandwidth that optimizes
# CV score.
# kNN is used to generate the seed adaptive bandwidth.

# kNN bandwidth
s.bw_method = "knn"
batch_size = 10000 # Batch size for KNN search
k = 10             # Numer of neighbors per batch
s.fit(N, scaling=scaling, batch_size=batch_size, k=k)
bw_knn = s.kde.bw

# MLCV optimization of previously calculated kNN bandwidth
s.bw_method = "mlcv"
N_cv = int(1E4)   # Use a smaller N to reduce computation times
seed = bw_knn[:N_cv] # Use kNN BW as seed (first N elements)
grid = np.logspace(-1,-0.4,10)
s.fit(N_cv, scaling=scaling, seed=seed, grid=grid)
bw_cv = s.kde.bw

# Extend MLCV optimization to full KNN BW
bw_knn_cv = bw_knn * bw_cv[0]/bw_knn[0] # Apply MLCV factor
dim = s.geom.dim
bw_knn_cv *= kds.kde.bw_silv(dim,len(bw_knn))/kds.kde.bw_silv(dim,len(bw_cv)) # Apply Silverman factor

# https://kdepy.readthedocs.io/en/latest/kernels.html, kernels can be plot as 
# from KDEpy import NaiveKDE
# for name, func in NaiveKDE._available_kernels.items():
#    x, y = NaiveKDE(kernel=name).fit([0]).evaluate()
#    plt.plot(x, y, label=name + (' (F)' if func.finite_support else ''))
# plt.grid(True, ls='--', zorder=-15); plt.legend();
# plt.show()

s = kds.KDSource(plist, geom, bw=bw_knn_cv,  kernel='epa') # Create new KDSource with full BW
s.fit(N=N, scaling=scaling)
s.plot_t
xmlfile = "source.xml"
s.save(xmlfile) # Save KDSource to XML file

os.system(f'time /home/caixx/git/cinema/external/KDSource/bin/kdtool resample source.xml -o res -n {resN}')
# os.system('mcpltool res.mcpl.gz -t res.txt')
os.system('pymcpltool --stats --pdf res.mcpl.gz')
os.system('mv mcpl.pdf mcpl_res.pdf')